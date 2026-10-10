# Lattice CLAA forgery attacks

Executable forgery demonstrations against two lattice-based certificateless
aggregate signature schemes for VANETs. Each program generates honest
signatures, aggregates them, and then builds a forged aggregate that the
scheme's aggregate verifier accepts without using any secret key.

| File | Scheme | Binary |
|---|---|---|
| `lattice.hpp` | Shared support code (mod-q arithmetic, matrices, SHA-256 hashing, Gaussian sampling, rejection test) | — |
| `lbclaa_attack.cpp` | LB-CLAA (two-equation check, attacker-chosen `h'`) | `lbclaa_play` |
| `lclaa_attack.cpp` | LCLAA (single equation `A·z = Σ(w + X·h3 + U·h4)`, solved mod 2^q) | `lclaa_play` |

## Requirements

- g++ (or clang++) with C++17 and `__int128` support (Linux / macOS; on Windows use WSL)
- OpenSSL development headers (`libssl-dev` on Ubuntu/Debian, `openssl` via Homebrew on macOS)

```bash
sudo apt install g++ libssl-dev
```

## Build

```bash
g++ -O2 -std=gnu++17 lbclaa_attack.cpp -o lbclaa_play -lcrypto
g++ -O2 -std=gnu++17 lclaa_attack.cpp  -o lclaa_play  -lcrypto
```

On macOS with Homebrew OpenSSL, add
`-I$(brew --prefix openssl)/include -L$(brew --prefix openssl)/lib`.

## Run

```bash
./lbclaa_play --small              # quick run (n=64, m=768)
./lbclaa_play                      # full parameters (n=512, m=6144)
./lbclaa_play --vehicles 16 --seed 42

./lclaa_play --small               # quick run (n=128)
./lclaa_play                       # full parameters (n=512, q=2^25)
./lclaa_play --n 256 --qbits 25 --vehicles 8 --seed 7
```

Check the result with `echo $?` right after each run.

| Option | lbclaa_play | lclaa_play |
|---|---|---|
| `--small` | n=k=64, m=768 | n=128 |
| `--vehicles N` | batch size (default 8) | batch size (default 8) |
| `--seed S` | RNG seed (hex or decimal) | RNG seed |
| `--n N` | — | lattice dimension (m = 2·n·qbits) |
| `--qbits B` | — | q = 2^B |

Full-size runs take a while (large matrix products); use `--small` first.

## Exit code

**lbclaa_play** returns `0` when the honest single signature verifies, the
honest aggregate verifies, and the forged aggregate passes both
verification equations.

**lclaa_play** returns `0` when the honest single signature verifies
(equation and `‖z‖ ≤ β`), the honest aggregate verifies, the forged
aggregate satisfies the aggregate equation, **and** the forged `z_agg` is
longer than `N·β`. The last condition shows the forgery is accepted only
because AggVerify has no norm bound on `z_agg`.

## How the attacks work

**LB-CLAA.** The attacker picks challenges `h'_i`, sets `d_agg = 0` and
`η'_agg = −Σ ID_i·h'_i` (likewise with `U_i` for the second equation).
Both aggregate equations then hold with no secret key. This relies on
the verifier taking `h_i` from the signature as-is, without recomputing it
from a hash over `η_i`.

**LCLAA.** LCLAA's AggVerify checks only
`A·z_agg = Σ(w_i + X_i·h3_i + U_i·h4_i)`. Individual Verify checks
`‖z‖ ≤ β`, but AggVerify puts no norm bound on `z_agg`. The attacker picks
`w'_i` and messages, computes the right-hand side, and solves the linear
system for `z_agg` mod 2^q. The forged `z_agg` is about 5–6× longer than
`N·β` (printed as "Forged z size … (gate …)"), and it is still accepted.
Adding `‖z_agg‖ ≤ N·β` to AggVerify blocks this forgery.

## .gitignore

Compiled binaries and Python caches are ignored via `.gitignore`.
To add it to the repo:

```bash
git add .gitignore README.md
git commit -m "Add README and .gitignore"
```

If a binary was already committed before the ignore rule existed, untrack it:

```bash
git rm --cached lbclaa_play lclaa_play
git commit -m "Stop tracking build outputs"
```
