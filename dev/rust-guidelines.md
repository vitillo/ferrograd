# Rust Best Practices

Consolidated from [Effective Rust](https://www.lurklurk.org/effective-rust/),
[Rust Design Patterns](https://rust-unofficial.github.io/patterns/), and
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines).

## Types & Type Safety

1. **Encode invariants as types** — use enums over booleans, `Option<T>` over sentinels, `Result<T,E>` over error codes
2. **Newtype pattern** — wrap primitives for domain safety (`Miles(f64)` vs bare `f64`)
3. **Custom enums over `bool`/`Option` params** — `Widget::new(Small, Round)` not `Widget::new(true, false)`
4. **Builder pattern for complex construction** — chainable setters, terminal `build()`
5. **Use `bitflags` crate for flag sets**, not enums
6. **Prefer `From`/`TryFrom`** — never implement `Into` directly; avoid `as` casts
7. **Accept the most general `Fn*` bound** — `FnOnce` > `FnMut` > `Fn`

## Error Handling

8. **Idiomatic errors with `thiserror`** — enum with one variant per failure, implement `Error + Send + Sync + 'static`
9. **Meaningful `Display` impls** — lowercase, no trailing punctuation
10. **Use `?` operator and `Option`/`Result` combinators** — `.map()`, `.and_then()`, `.unwrap_or()` over verbose `match`
11. **No `.unwrap()` in non-test code** — use `expect()` with context or propagate errors
12. **Document failure modes** — `# Errors` for `Result`, `# Panics` for panic conditions

## Ownership & Borrowing

13. **Accept borrowed types for arguments** — `&str` not `&String`, `&[T]` not `&Vec<T>`
14. **Let caller control allocation** — accept owned when you need ownership, borrows when you don't
15. **Use `mem::take`/`mem::replace`** to move out of mutable refs without cloning
16. **Don't clone to satisfy the borrow checker** — restructure code instead
17. **Temporary mutability** — `let mut x = ...; setup(); let x = x;`
18. **Return consumed args on error** — so caller can retry without cloning

## Traits & Generics

19. **Derive standard traits eagerly** — `Clone, Debug, PartialEq, Eq, Hash, Default` on all applicable types
20. **Implement `Default`** for types with sensible zero-values
21. **Implement `Display`** for user-facing types
22. **Implement `FromIterator` and `Extend`** on collections
23. **Use generics by default** — trait objects only for heterogeneous collections or binary size
24. **Default implementations** to minimize required trait methods
25. **Object safety** — add `where Self: Sized` to generic helpers to keep traits object-safe

## Iterators & Functional Style

26. **Iterator transforms over explicit loops** — `.iter().filter().map()` with purpose-built consumers (`sum`, `find`, `any`, `collect`)
27. **`format!` for string concatenation** — reserve manual `push_str` for hot paths
28. **Treat `Option` as zero-or-one iterator** — use `.chain()`, `.extend()`
29. **Collections provide `iter()`, `iter_mut()`, `into_iter()`**

## API Design

30. **Minimize visibility** — default to private, use `pub(crate)` before bare `pub`
31. **No `get_` prefix on simple accessors** — `fn first()` not `fn get_first()`
32. **Conversion naming** — `as_` (free borrow), `to_` (expensive), `into_` (consuming)
33. **Methods over free functions** when there's a clear receiver
34. **Return values via tuples/structs**, not out-parameters
35. **Operator overloads only for genuine math semantics**
36. **`Deref`/`DerefMut` only on smart pointers** — not for fake inheritance
37. **Expose intermediate results** — return useful info on failure (insertion points, byte offsets)
38. **Use `#[non_exhaustive]`** on public structs/enums for future extensibility

## Code Quality

39. **Keep Clippy pedantic warning-free** — suppress specific lints with `#[allow(...)]`
40. **Avoid wildcard imports** — explicit `use` except in test modules
41. **Avoid `unsafe`** — search std/crates.io first; isolate in thin wrapper modules with `// SAFETY:` comments
42. **Macros as last resort** — functions and generics first
43. **Compose structs** — split large structs so borrow checker allows disjoint mutable borrows
44. **Prefer small, focused modules**
45. **Avoid over-optimization** — favor owned types for simplicity, optimize after profiling

## Documentation

46. **Doc comments on all public items** with examples
47. **Use `?` in doc examples**, not `unwrap`
48. **Hyperlink related types** in doc prose
49. **`# Errors`, `# Panics`, `# Safety` sections** where applicable

## Future-Proofing

50. **Sealed traits** when you need to add methods without breaking downstream
51. **Private struct fields by default** — getters/setters preserve invariants
52. **No trait bounds on struct definitions** — let `#[derive]` handle it
53. **Wrap complex internal types in newtypes** to hide implementation details
