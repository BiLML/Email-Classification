# 🔐 Information Security (ATTT) Knowledge Base

> Covers: Cryptographic algorithm implementations (from scratch in Python)
> Source: Academic ATTT thesis project (April 2026)

---

## AES-128-CBC (Symmetric Encryption)

### Overview
- Block size: **128 bits (16 bytes)**
- Key size: **128 bits**
- Rounds: **10 rounds** of transformation
- Mode: **CBC (Cipher Block Chaining)**

### Core Transformations per Round

| Transformation | Description |
|---------------|-------------|
| **SubBytes** | Substitute each byte via S-Box lookup (non-linear, GF(2⁸)) |
| **ShiftRows** | Cyclically shift rows: row0→0, row1→1, row2→2, row3→3 bytes |
| **MixColumns** | Multiply each column by matrix in GF(2⁸) — diffusion |
| **AddRoundKey** | XOR state with round key |
| **Key Expansion** | Expand 128-bit key → 11 round keys (RotWord + SubWord + Rcon) |

### Round Structure
```
Plaintext → AddRoundKey(Key₀)
          → 9 rounds: SubBytes → ShiftRows → MixColumns → AddRoundKey
          → Final:    SubBytes → ShiftRows → AddRoundKey(Key₁₀)
          → Ciphertext
```

### CBC Mode
```
Encryption: Cᵢ = AES_Encrypt(Pᵢ ⊕ Cᵢ₋₁),  C₀ = IV
Decryption: Pᵢ = AES_Decrypt(Cᵢ) ⊕ Cᵢ₋₁,   C₀ = IV
```
- IV: random 16 bytes, prepended to ciphertext
- File: `d:\ATTT\AES\AES.py`

---

## RSA (Asymmetric Encryption)

### Key Generation
1. Choose two large primes: `p`, `q` (via `random_prime.py`)
2. Compute `n = p * q`, `φ(n) = (p-1)(q-1)`
3. Choose `e` such that `gcd(e, φ(n)) = 1` (commonly `e = 65537`)
4. Compute `d = e⁻¹ mod φ(n)` (modular inverse)
5. Public key: `(e, n)`, Private key: `(d, n)`

### Encrypt / Decrypt
```
Encrypt: C = Mᵉ mod n
Decrypt: M = Cᵈ mod n
```

### Files
- `d:\ATTT\RSA\RSA.py`
- `d:\ATTT\RSA\random_prime.py`

---

## SHA-256 (Cryptographic Hash)

- Output: **256-bit (32-byte)** digest
- One-way: infeasible to reverse
- Used as basis for digital signatures
- File: `d:\ATTT\SHA256\SHA256.py`

---

## Digital Signature (RSA-based)

### Sign
```
1. Hash message:  h = SHA256(M)
2. Sign:          S = hᵈ mod n    (private key)
```

### Verify
```
1. Compute:       h' = Sᵉ mod n  (public key)
2. Hash message:  h  = SHA256(M)
3. Verify:        h == h'  →  authentic
```

- File: `d:\ATTT\Chữ ký số\DigitalSignature.py`

---

## Python Environment Setup (ATTT Project)
```powershell
# Activate virtual environment
.env/Scripts/Activate.ps1

# Typical structure
d:\ATTT\
├── AES\AES.py
├── RSA\RSA.py
│   RSA\random_prime.py
├── SHA256\SHA256.py
└── Chữ ký số\DigitalSignature.py
```
