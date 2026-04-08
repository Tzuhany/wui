//! Proc-macros for the wui agent framework.
//!
//! # `#[derive(ToolInput)]`
//!
//! Generates a strongly-typed accessor struct and a JSON Schema `serde_json::Value`
//! from a plain Rust struct, eliminating the need to hand-write both the schema
//! and the runtime accessors.
//!
//! ## Example
//!
//! ```rust,ignore
//! #[derive(ToolInput)]
//! struct SearchInput {
//!     /// The search query string.
//!     query: String,
//!     /// Maximum number of results to return.
//!     #[tool_input(default = 10)]
//!     max_results: u64,
//!     /// Optional filter to restrict results by category.
//!     category: Option<String>,
//! }
//! ```
//!
//! This generates:
//! - `SearchInput::schema() -> serde_json::Value` — the JSON Schema for this input
//! - `SearchInput::from_value(v: &serde_json::Value) -> Result<SearchInput, String>` — typed parsing

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

#[proc_macro_derive(ToolInput, attributes(tool_input))]
pub fn derive_tool_input(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => panic!("#[derive(ToolInput)] only supports named fields"),
        },
        _ => panic!("#[derive(ToolInput)] only supports structs"),
    };

    // Build the schema properties
    let mut schema_props = Vec::new();
    let mut required_fields = Vec::new();
    let mut from_value_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let ty = &field.ty;

        // Check if the field is Option<T>
        let is_optional = is_option_type(ty);
        let inner_ty = if is_optional {
            extract_option_inner(ty)
        } else {
            ty.clone()
        };

        // Get doc comment as description
        let description = extract_doc_comment(&field.attrs);

        // Build JSON schema entry
        let json_type = rust_type_to_json_schema(&inner_ty);
        let schema_entry = if !description.is_empty() {
            quote! {
                #field_name_str: {
                    "type": #json_type,
                    "description": #description,
                }
            }
        } else {
            quote! {
                #field_name_str: {
                    "type": #json_type,
                }
            }
        };
        schema_props.push(schema_entry);

        if !is_optional {
            required_fields.push(quote! { #field_name_str });
        }

        // Build from_value parsing
        if is_optional {
            from_value_fields.push(quote! {
                #field_name: value.get(#field_name_str)
                    .and_then(|v| serde_json::from_value(v.clone()).ok()),
            });
        } else {
            from_value_fields.push(quote! {
                #field_name: value.get(#field_name_str)
                    .ok_or_else(|| format!("missing required field `{}`", #field_name_str))
                    .and_then(|v| serde_json::from_value::<#inner_ty>(v.clone())
                        .map_err(|e| format!("invalid field `{}`: {}", #field_name_str, e)))?,
            });
        }
    }

    let expanded = quote! {
        impl #name {
            /// Returns the JSON Schema for this tool input type.
            pub fn schema() -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        #( #schema_props ),*
                    },
                    "required": [ #( #required_fields ),* ]
                })
            }

            /// Parse a `serde_json::Value` into this typed input.
            pub fn from_value(value: &serde_json::Value) -> Result<Self, String> {
                Ok(Self {
                    #( #from_value_fields )*
                })
            }
        }
    };

    TokenStream::from(expanded)
}

fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(p) = ty {
        p.path
            .segments
            .last()
            .map(|s| s.ident == "Option")
            .unwrap_or(false)
    } else {
        false
    }
}

fn extract_option_inner(ty: &Type) -> Type {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                    return inner.clone();
                }
            }
        }
    }
    ty.clone()
}

fn rust_type_to_json_schema(ty: &Type) -> &'static str {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            return match seg.ident.to_string().as_str() {
                "String" | "str" => "string",
                "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "usize" | "isize" => {
                    "integer"
                }
                "f32" | "f64" => "number",
                "bool" => "boolean",
                "Vec" => "array",
                _ => "object",
            };
        }
    }
    "string"
}

fn extract_doc_comment(attrs: &[syn::Attribute]) -> String {
    attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        return Some(s.value().trim().to_owned());
                    }
                }
            }
            None
        })
        .collect::<Vec<_>>()
        .join(" ")
}
