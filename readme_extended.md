## SIMD library (with SSE and AVX)
They are stored in "simd" folder (Only nessacary ones).
The whole SSE/AVX library is stored in "simd_full" folder.

It is not possible to just use "#include <immintrin.h>" in Intel SGX. You should include them separately.
Use in SGX "#include <mmintrin.h">" instead "#include "mmintrin.h"

