---
name: release
description: Run the full wui release workflow — bump version, update CHANGELOG, publish all crates.
---

You are running the wui release workflow. Follow these steps carefully:

## Pre-flight checks

1. Verify the working tree is clean: `git status`
2. Verify you are on the `master` branch: `git branch --show-current`
3. Run the full test suite: `cargo test --workspace --features full`
4. Check for clippy warnings: `cargo clippy --workspace --features full -- -D warnings`

If any check fails, stop and report the issue.

## Determine the version bump

Look at commits since the last tag (`git log $(git describe --tags --abbrev=0)..HEAD --oneline`):
- Any `feat:` commit → minor bump
- Only `fix:` or `refactor:` commits → patch bump
- Any `BREAKING CHANGE` in commit body → major bump

Ask the user to confirm the proposed bump level before proceeding.

## Execute the release

1. Update version in root `Cargo.toml` `[workspace.package]` section
2. Run `cargo build --workspace` to update `Cargo.lock`
3. Generate CHANGELOG entry: `git cliff --unreleased --tag v<NEW_VERSION> --prepend CHANGELOG.md`
4. Commit: `git commit -am "chore: release v<NEW_VERSION>"`
5. Tag: `git tag -a "v<NEW_VERSION>" -m "v<NEW_VERSION>"`
6. Publish crates in dependency order:
   - `cargo publish -p wui-core`
   - Wait ~15s for crates.io to index
   - `cargo publish -p wui`
   - `cargo publish -p wui-mcp`
   - `cargo publish -p wui-memory`
   - `cargo publish -p wui-observe`
   - `cargo publish -p wui-workflow`
   - `cargo publish -p wui-spawn`
   - `cargo publish -p wui-skills`
7. Push: `git push && git push --tags`

## Post-release

Confirm all packages appear on crates.io.
Report the final version and tag to the user.
