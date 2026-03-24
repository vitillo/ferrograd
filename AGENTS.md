# ferrograd

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
- Compare IR at each pipeline stage by running the same op in both:
  - rustgrad: `DEBUG=3 cargo test <test_name> -- --nocapture` shows tensor graph → rangeify before/after → symbolic before/after
  - tinygrad: `/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/bin/python3.14` with `DEBUG=4` or inspect `r.uop.toposort()` for the lazy graph
  - Check that the tensor-level graph structure and kernel-level IR are not diverging substantially
  - To inspect tinygrad's kernel IR after rangeify (the scheduled AST):
    ```python
    cd ~/projects/tinygrad
    python3.14 -c "
    from tinygrad import Tensor
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2,3)
    r = a.sum(axis=1)
    sched, _ = Tensor.schedule_with_vars(r)
    for i, si in enumerate(sched):
        print(f'--- kernel {i} ---')
        for j, u in enumerate(si.ast.toposort()):
            srcs = ', '.join([f'%{k}' for k, s in enumerate(si.ast.toposort()) if any(s is x for x in u.src)])
            arg_str = f'  arg={u.arg}' if u.arg is not None else ''
            print(f'  %{j} = {u.op} {u.dtype} ({srcs}){arg_str}')
    "
    ```
  - python3.14 is at: `/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/bin/python3.14`

## Documentation
- `missing_docs` lint is enabled -- all public items need docs
- Explain *why*, not just *what* -- this is an educational project
- Reference tinygrad equivalents where relevant

## Commits
- Narrative form, focus on *why* the change was made, not a bullet list of *what*
- No Codex attribution in commit messages

## Workflow
- Pre-commit hook runs clippy + tests automatically
- Build with `cargo clippy` (superset of `cargo build`)
