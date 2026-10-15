# Third-party notices

The Model 8 denoiser and diffusion loop were independently adapted for PBISC
from the architecture and algorithms in the following MIT-licensed project:

- Project: `PhilSid/sparse-data-diffusion`
- Repository: https://github.com/PhilSid/sparse-data-diffusion
- Reference commit: `1d5423c13477f2a52269a6f79113b45c6af5510f`

The PBISC adaptation adds conditional bulk/depth inputs, a three-state
measurement contract, measured-only BCE/L2 losses, strict data provenance,
and native PBISC E2 streaming/checkpointing. It does not vendor the upstream
repository.

## Upstream license

MIT License

Copyright (c) 2026 PhilSid

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
