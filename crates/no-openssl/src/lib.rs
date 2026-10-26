//! A guard, not a library: no OpenSSL anywhere in this workspace
//! (rule 5 of the agora repo's CLAUDE.md, rustls only).
//!
//! This crate's manifest declares `links = "openssl"`, as `openssl-sys` does.
//! Cargo allows only one package per `links` value in a dependency graph, so
//! if any dependency, under any feature or target, pulls `openssl-sys` in,
//! every cargo command in the workspace fails at dependency resolution, before
//! anything compiles. The error names `openssl-sys` and this crate.
//!
//! Never "fix" that error by removing this crate or its `links` key. Find the
//! dependency that brought OpenSSL in (`cargo tree -i openssl-sys` with the
//! guard removed locally) and switch it to its rustls feature.
//!
//! One gap: `openssl-sys` 0.9.28 to 0.9.30 were published before the
//! crates.io index recorded `links`, so a requirement loose enough to reach
//! them (a bare `openssl-sys = "0.9"`) resolves to one of those instead of
//! failing. The usual ways in (`native-tls`, `openssl` 0.10) require newer
//! releases, and those three predate OpenSSL 3; CI's `cargo tree` check
//! catches the rest.
