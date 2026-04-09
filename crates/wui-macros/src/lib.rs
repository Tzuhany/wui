// ============================================================================
// wui-macros — procedural macros for the wui agent framework.
//
// Currently provides:
//   #[derive(ToolInput)] — generate schema() and from_value() for tool inputs.
// ============================================================================

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Field, Fields, Lit, Meta, Type};

// ── #[derive(ToolInput)] ──────────────────────────────────────────────────────

/// Derive `schema()` and `from_value()` for a tool input struct.
///
/// # What it generates
///
/// For each annotated struct, the macro emits:
///
/// ```rust,ignore
/// impl MyInput {
///     /// Returns a JSON Schema describing the tool's input parameters.
///     pub fn schema() -> ::serde_json::Value { ... }
///
///     /// Deserialize the struct from a raw `serde_json::Value`.
///     ///
///     /// Returns `Ok(Self)` on success, or an `Err(String)` describing which
///     /// field was missing or invalid.
///     pub fn from_value(value: &::serde_json::Value) -> Result<Self, String> { ... }
/// }
/// ```
///
/// # Field rules
///
/// | Field type            | Treatment                                         |
/// |-----------------------|---------------------------------------------------|
/// | `T` (non-Option)      | Required field — error if absent or wrong type    |
/// | `Option<T>`           | Optional field — `None` when absent               |
///
/// Doc comments on fields are extracted and included in the JSON Schema as
/// `"description"` properties.
///
/// # Type mapping
///
/// The JSON Schema `"type"` is inferred from the Rust type:
///
/// | Rust type             | JSON Schema type |
/// |-----------------------|------------------|
/// | `String` / `&str`     | `"string"`       |
/// | `bool`                | `"boolean"`      |
/// | `u8`…`u64`, `usize`   | `"integer"`      |
/// | `i8`…`i64`, `isize`   | `"integer"`      |
/// | `f32`, `f64`          | `"number"`       |
/// | `Vec`                 | `"array"`        |
/// | `HashMap` / `BTreeMap` / `IndexMap` | `"object"` |
/// | `Value` (serde_json)  | `"object"`       |
/// | anything else         | `"string"` (fallback) |
///
/// # Prerequisites
///
/// The consuming crate must have `serde_json` as a direct dependency. Generated
/// code uses fully-qualified `::serde_json::…` paths to avoid ambiguity.
///
/// # Example
///
/// ```rust,ignore
/// use wui_macros::ToolInput;
///
/// #[derive(ToolInput)]
/// struct SearchInput {
///     /// The search query.
///     query: String,
///     /// Maximum number of results to return.
///     max_results: Option<u64>,
/// }
///
/// // Generated:
/// // SearchInput::schema()      → serde_json::Value (JSON Schema object)
/// // SearchInput::from_value()  → Result<SearchInput, String>
/// ```
#[proc_macro_derive(ToolInput, attributes(tool))]
pub fn derive_tool_input(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_tool_input(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

// ── Expansion ─────────────────────────────────────────────────────────────────

fn expand_tool_input(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "#[derive(ToolInput)] only supports structs with named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new(
                Span::call_site(),
                "#[derive(ToolInput)] only supports structs",
            ))
        }
    };

    let mut schema_props = Vec::new();
    let mut required_fields: Vec<String> = Vec::new();
    let mut from_value_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let description = extract_doc_comment(field);
        let (is_optional, inner_ty) = extract_option_inner(&field.ty);
        let json_type = rust_type_to_json_type(inner_ty);

        // Build the schema property.
        let prop = if let Some(desc) = description {
            quote! {
                #field_name_str: {
                    "type": #json_type,
                    "description": #desc
                }
            }
        } else {
            quote! {
                #field_name_str: {
                    "type": #json_type
                }
            }
        };
        schema_props.push(prop);

        // Track required fields.
        if !is_optional {
            required_fields.push(field_name_str.clone());
        }

        // Build the from_value field extraction.
        let field_extraction = if is_optional {
            quote! {
                #field_name: value.get(#field_name_str)
                    .and_then(|v| ::serde_json::from_value(v.clone()).ok())
            }
        } else {
            let missing_msg = format!("missing required field `{field_name_str}`");
            let invalid_fmt = format!("invalid field `{field_name_str}`: {{e}}");
            quote! {
                #field_name: value.get(#field_name_str)
                    .ok_or_else(|| #missing_msg.to_string())
                    .and_then(|v| ::serde_json::from_value(v.clone())
                        .map_err(|e| format!(#invalid_fmt)))?
            }
        };
        from_value_fields.push(field_extraction);
    }

    let required_array = if required_fields.is_empty() {
        quote! { ::serde_json::json!([]) }
    } else {
        quote! { ::serde_json::json!([ #(#required_fields),* ]) }
    };

    let expanded = quote! {
        impl #name {
            /// Returns a JSON Schema object describing this tool's input parameters.
            pub fn schema() -> ::serde_json::Value {
                ::serde_json::json!({
                    "type": "object",
                    "properties": {
                        #(#schema_props),*
                    },
                    "required": #required_array
                })
            }

            /// Deserialise from a raw `serde_json::Value`.
            ///
            /// Returns `Ok(Self)` on success, or `Err(String)` naming the
            /// struct and the missing or invalid field.
            pub fn from_value(value: &::serde_json::Value) -> Result<Self, String> {
                Ok(Self {
                    #(#from_value_fields),*
                })
            }
        }

        impl ::wui_core::tool::ToolArgs for #name {
            fn schema() -> ::serde_json::Value {
                #name::schema()
            }

            fn parse(value: ::serde_json::Value) -> Result<Self, ::wui_core::tool::ToolInputError> {
                #name::from_value(&value).map_err(|msg| {
                    ::wui_core::tool::ToolInputError::new(msg)
                })
            }
        }
    };

    Ok(expanded)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract the text of all `///` doc comments on a field, concatenated.
fn extract_doc_comment(field: &Field) -> Option<String> {
    let lines: Vec<String> = field
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => Some(&nv.value),
            _ => None,
        })
        .filter_map(|expr| match expr {
            syn::Expr::Lit(expr_lit) => match &expr_lit.lit {
                Lit::Str(s) => Some(s.value().trim().to_string()),
                _ => None,
            },
            _ => None,
        })
        .collect();

    (!lines.is_empty()).then(|| lines.join(" "))
}

/// If `ty` is `Option<T>`, returns `(true, T)`. Otherwise returns `(false, ty)`.
fn extract_option_inner(ty: &Type) -> (bool, &Type) {
    let Type::Path(tp) = ty else {
        return (false, ty);
    };
    let seg = tp.path.segments.last().filter(|s| s.ident == "Option");
    let Some(seg) = seg else { return (false, ty) };
    let syn::PathArguments::AngleBracketed(ab) = &seg.arguments else {
        return (false, ty);
    };
    match ab.args.first() {
        Some(syn::GenericArgument::Type(inner)) => (true, inner),
        _ => (false, ty),
    }
}

/// Map a Rust type to a JSON Schema type string.
fn rust_type_to_json_type(ty: &Type) -> &'static str {
    let Type::Path(tp) = ty else { return "string" };
    let Some(seg) = tp.path.segments.last() else {
        return "string";
    };
    match seg.ident.to_string().as_str() {
        "String" | "str" => "string",
        "bool" => "boolean",
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
        | "isize" => "integer",
        "f32" | "f64" => "number",
        "Vec" => "array",
        "HashMap" | "BTreeMap" | "IndexMap" | "Value" => "object",
        _ => "string",
    }
}
