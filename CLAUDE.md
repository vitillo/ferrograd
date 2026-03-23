# rustgrad

Educational tensor compiler in Rust, inspired by tinygrad. Plan: `dev/PLAN.md`

## Code style
- Follow modern Rust conventions
- Use `thiserror` for typed errors, not `String`
- Clippy pedantic is on -- keep it warning-free

## Tests
- Arrange / Act / Assert structure with section comments
- Test both happy paths and error cases

## Tinygrad alignment
- Every IR op, type, and abstraction must have a tinygrad equivalent -- don't invent concepts that don't exist upstream
- Reference tinygrad source: `~/projects/tinygrad`

## Documentation
- `missing_docs` lint is enabled -- all public items need docs
- Explain *why*, not just *what* -- this is an educational project
- Reference tinygrad equivalents where relevant

## Commits
- Narrative form, focus on *why* the change was made, not a bullet list of *what*
- No Claude attribution in commit messages

## Workflow
- Pre-commit hook runs clippy + tests automatically
- Build with `cargo clippy` (superset of `cargo build`)
