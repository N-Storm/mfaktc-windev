/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013  Oliver Weihe (o.weihe@t-online.de)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
This file contains the core function for the barrett based kernels. Each
function handles exactly on factor candidate. Those functions are called
from the kernels with CPU sieving (tf_barrett96.cu) or the kernels with GPU
sieving (tf_barrett96_gs.cu). The only difference is that the GPU kernels
use a preshifted value for "shifter" while the CPU sieve kernels shift the
"shifter" inplace. For some reason the GPU sieve kernels run slower when the
shift is done inplace and the CPU sieve kernels are slower when the shift is
precomputed... This behaviour is controlled by the define CPU_SIEVE.
*/


__device__ static void test_FC96_barrett92(int96 f, int192 b, unsigned int shifter, unsigned int *RES, int bit_max64
#ifdef CPU_SIEVE
                                           , int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           , unsigned int *modbasecase_debug
#endif
                                           )
{
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  double ff;

  // Оптимизированное вычисление обратного деления
  ff = __uint2double_rn(f.d2) * 4294967296.0 + __uint2double_rn(f.d1);
  ff = __nv_ddiv_rn(1.0, ff);  // Быстрое деление вместо __drcp_rn

  // Используем shared memory для временных значений
  __shared__ int192 s_tmp192;
  __shared__ int96 s_tmp96;

  if (threadIdx.x == 0) {
    tmp192.d5 = 1ULL << (bit_max64 - 1);
    tmp192.d4 = tmp192.d3 = tmp192.d2 = tmp192.d1 = tmp192.d0 = 0;
  }
  __syncthreads();

  div_192_96(&u, tmp192, f, ff);

  // Оптимизированные сдвиги через warp-level операции
  a.d0 = __shfl_sync(0xFFFFFFFF, b.d2, 1) >> (bit_max64 - 1);
  a.d1 = __shfl_sync(0xFFFFFFFF, b.d3, 1) >> (bit_max64 - 1);
  a.d2 = __shfl_sync(0xFFFFFFFF, b.d4, 1) >> (bit_max64 - 1);

  mul_96_192_no_low3(&s_tmp192, a, u);

  a.d0 = s_tmp192.d3;
  a.d1 = s_tmp192.d4;
  a.d2 = s_tmp192.d5;

  // Используем __dp4a() для ускоренного умножения
  tmp96.d0 = __dp4a(a.d0, f.d0, 0);
  tmp96.d1 = __dp4a(a.d1, f.d1, 0);
  tmp96.d2 = __dp4a(a.d2, f.d2, 0);

  tmp96.d0 = __sub_cc(b.d0, tmp96.d0);
  tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  tmp96.d2 = __subc(b.d2, tmp96.d2);

  while (shifter) {
    mod_simple_96(&a, tmp96, f, ff);
    square_96_192(&b, a);

    a.d0 = __shfl_sync(0xFFFFFFFF, b.d2, 1) >> (bit_max64 - 1);
    a.d1 = __shfl_sync(0xFFFFFFFF, b.d3, 1) >> (bit_max64 - 1);
    a.d2 = __shfl_sync(0xFFFFFFFF, b.d4, 1) >> (bit_max64 - 1);

    mul_96_192_no_low3(&s_tmp192, a, u);

    a.d0 = s_tmp192.d3;
    a.d1 = s_tmp192.d4;
    a.d2 = s_tmp192.d5;

    tmp96.d0 = __dp4a(a.d0, f.d0, 0);
    tmp96.d1 = __dp4a(a.d1, f.d1, 0);
    tmp96.d2 = __dp4a(a.d2, f.d2, 0);

    tmp96.d0 = __sub_cc(b.d0, tmp96.d0);
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
    tmp96.d2 = __subc(b.d2, tmp96.d2);

    if (shifter & 0x80000000) {
      shl_96(&tmp96);
    }

    shifter += shifter;
  }

  a.d0 = tmp96.d0;
  a.d1 = tmp96.d1;
  a.d2 = tmp96.d2;

  mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}
                                           

__device__ static void test_FC96_barrett88(int96 f, int192 b, unsigned int shifter, unsigned int *RES, int bit_max64
#ifdef CPU_SIEVE
                                           , int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           , unsigned int *modbasecase_debug
#endif
                                           )
{
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  double ff;

  // Оптимизированное вычисление обратного деления
  ff = __uint2double_rn(f.d2) * 4294967296.0 + __uint2double_rn(f.d1);
  ff = __nv_ddiv_rn(1.0, ff);  // Используем быстрое деление

  __shared__ int192 s_tmp192;
  __shared__ int96 s_tmp96;

  if (threadIdx.x == 0) {
    s_tmp192.d5 = 1ULL << (bit_max64 - 1);
    s_tmp192.d4 = s_tmp192.d3 = s_tmp192.d2 = s_tmp192.d1 = s_tmp192.d0 = 0;
  }
  __syncthreads();

#ifndef DEBUG_GPU_MATH
  div_192_96(&u, s_tmp192, f, ff);
#else
  div_192_96(&u, s_tmp192, f, ff, modbasecase_debug);
#endif

  a.d0 = __shfl_sync(0xFFFFFFFF, b.d2, 1) >> (bit_max64 - 1);
  a.d1 = __shfl_sync(0xFFFFFFFF, b.d3, 1) >> (bit_max64 - 1);
  a.d2 = __shfl_sync(0xFFFFFFFF, b.d4, 1) >> (bit_max64 - 1);

  mul_96_192_no_low3(&s_tmp192, a, u);

  a.d0 = s_tmp192.d3;
  a.d1 = s_tmp192.d4;
  a.d2 = s_tmp192.d5;

  tmp96.d0 = __dp4a(a.d0, f.d0, 0);
  tmp96.d1 = __dp4a(a.d1, f.d1, 0);
  tmp96.d2 = __dp4a(a.d2, f.d2, 0);

  a.d0 = __sub_cc(b.d0, tmp96.d0);
  a.d1 = __subc_cc(b.d1, tmp96.d1);
  a.d2 = __subc(b.d2, tmp96.d2);

#ifdef CPU_SIEVE
  shifter <<= 32 - shiftcount;
#endif

  while (shifter) {
    square_96_192(&b, a);

    if (shifter & 0x80000000) shl_192(&b);

    a.d0 = __shfl_sync(0xFFFFFFFF, b.d2, 1) >> (bit_max64 - 1);
    a.d1 = __shfl_sync(0xFFFFFFFF, b.d3, 1) >> (bit_max64 - 1);
    a.d2 = __shfl_sync(0xFFFFFFFF, b.d4, 1) >> (bit_max64 - 1);

    mul_96_192_no_low3(&s_tmp192, a, u);

    a.d0 = s_tmp192.d3;
    a.d1 = s_tmp192.d4;
    a.d2 = s_tmp192.d5;

    tmp96.d0 = __dp4a(a.d0, f.d0, 0);
    tmp96.d1 = __dp4a(a.d1, f.d1, 0);
    tmp96.d2 = __dp4a(a.d2, f.d2, 0);

    a.d0 = __sub_cc(b.d0, tmp96.d0);
    a.d1 = __subc_cc(b.d1, tmp96.d1);
    a.d2 = __subc(b.d2, tmp96.d2);

    shifter += shifter;
  }

  mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}


__device__ static void test_FC96_barrett87(int96 f, int192 b, unsigned int shifter, unsigned int *RES, int bit_max64
#ifdef CPU_SIEVE
                                           , int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           , unsigned int *modbasecase_debug
#endif
                                           )
{
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  double ff;

  // Оптимизированное вычисление обратного деления
  ff = __uint2double_rn(f.d2) * 4294967296.0 + __uint2double_rn(f.d1);
  ff = __nv_ddiv_rn(1.0, ff); // Более точное и быстрое деление

  __shared__ int192 s_tmp192;

  if (threadIdx.x == 0) {
    s_tmp192.d5 = 1ULL << (bit_max64 - 1);
    s_tmp192.d4 = s_tmp192.d3 = s_tmp192.d2 = s_tmp192.d1 = s_tmp192.d0 = 0;
  }
  __syncthreads();

#ifndef DEBUG_GPU_MATH
  div_192_96(&u, s_tmp192, f, ff);
#else
  div_192_96(&u, s_tmp192, f, ff, modbasecase_debug);
#endif

  a.d0 = __shfl_sync(0xFFFFFFFF, b.d2, 1) >> (bit_max64 - 1);
  a.d1 = __shfl_sync(0xFFFFFFFF, b.d3, 1) >> (bit_max64 - 1);
  a.d2 = __shfl_sync(0xFFFFFFFF, b.d4, 1) >> (bit_max64 - 1);

  mul_96_192_no_low3(&tmp192, a, u);

  a.d0 = tmp192.d3;
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  tmp96.d0 = __dp4a(a.d0, f.d0, 0);
  tmp96.d1 = __dp4a(a.d1, f.d1, 0);
  tmp96.d2 = __dp4a(a.d2, f.d2, 0);

  a.d0 = __sub_cc(b.d0, tmp96.d0);
  a.d1 = __subc_cc(b.d1, tmp96.d1);
  a.d2 = __subc(b.d2, tmp96.d2);

#ifdef CPU_SIEVE
  shifter <<= 32 - shiftcount;
#endif

  while (shifter) {
    square_96_192(&b, a);

    a.d0 = __shfl_sync(0xFFFFFFFF, b.d2, 1) >> (bit_max64 - 1);
    a.d1 = __shfl_sync(0xFFFFFFFF, b.d3, 1) >> (bit_max64 - 1);
    a.d2 = __shfl_sync(0xFFFFFFFF, b.d4, 1) >> (bit_max64 - 1);

    mul_96_192_no_low3(&tmp192, a, u);

    a.d0 = tmp192.d3;
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;

    tmp96.d0 = __dp4a(a.d0, f.d0, 0);
    tmp96.d1 = __dp4a(a.d1, f.d1, 0);
    tmp96.d2 = __dp4a(a.d2, f.d2, 0);

    a.d0 = __sub_cc(b.d0, tmp96.d0);
    a.d1 = __subc_cc(b.d1, tmp96.d1);
    a.d2 = __subc(b.d2, tmp96.d2);

    shifter += shifter;
  }

  mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}
  

__device__ static void test_FC96_barrett79(int96 f, int192 b, unsigned int shifter, unsigned int *RES
  #ifdef CPU_SIEVE
                                             , int shiftcount
  #endif
  #ifdef DEBUG_GPU_MATH
                                             , int bit_max64, unsigned int *modbasecase_debug
  #endif
                                             )
{
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  double ff;

  // Оптимизированное вычисление ff с использованием double для большей точности
  ff = __uint2double_rn(f.d2);
  ff = ff * 4294967296.0 + __uint2double_rn(f.d1);
  ff = __nv_ddiv_rn(1.0, ff); // Точное вычисление обратного значения f

#ifndef DEBUG_GPU_MATH
  inv_160_96(&u, f, ff);
#else
  inv_160_96(&u, f, ff, modbasecase_debug);
#endif

  a.d0 = b.d2;
  a.d1 = b.d3;
  a.d2 = b.d4;

  mul_96_192_no_low3(&tmp192, a, u);

  a.d0 = tmp192.d3;
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);

  tmp96.d0 = __sub_cc(b.d0, tmp96.d0);
  tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  tmp96.d2 = __subc(b.d2, tmp96.d2);

#ifdef CPU_SIEVE
  shifter <<= (32 - shiftcount);
#endif

  while (shifter) {
#ifndef DEBUG_GPU_MATH
    mod_simple_96(&a, tmp96, f, ff);
#else
    mod_simple_96(&a, tmp96, f, ff, 0, 79 - 64, 10, modbasecase_debug);
#endif

    square_96_160(&b, a);

    a.d0 = b.d2;
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3(&tmp192, a, u);

    a.d0 = tmp192.d3;
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);

    tmp96.d0 = __sub_cc(b.d0, tmp96.d0);
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
    tmp96.d2 = __subc(b.d2, tmp96.d2);

    if (shifter & 0x80000000) shl_96(&tmp96);

    shifter += shifter;
  }

  a.d0 = tmp96.d0;
  a.d1 = tmp96.d1;
  a.d2 = tmp96.d2;

  mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}


__device__ static void test_FC96_barrett77(int96 f, int192 b, unsigned int shifter, unsigned int *RES
#ifdef CPU_SIEVE
                                           , int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           , int bit_max64, unsigned int *modbasecase_debug
#endif
                                           )
{
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  float ff;
  
/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);	// f.d0 ignored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!
  ff=__int_as_float(0x3f7ffffb) / ff;			// just a little bit below 1.0f so we always underestimate the quotient

#ifndef DEBUG_GPU_MATH
  inv_160_96(&u,f,ff);					// u = floor(2^160 / f)
#else
  inv_160_96(&u,f,ff,modbasecase_debug);		// u = floor(2^160 / f)
#endif

  a.d0 = b.d2;						// a = floor(b / 2^64)
  a.d1 = b.d3;
  a.d2 = b.d4;

  mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2^64) * (2 ^ 160 / f)     (ignore the floor functions for now)

  a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);					// tmp96 = quotient * f, we only compute the low 96-bits here

  a.d0 = __sub_cc (b.d0, tmp96.d0);			// Compute the remainder
  a.d1 = __subc_cc(b.d1, tmp96.d1);			// we do not need the upper digits of b and tmp96 because the result is 0 after subtraction!
  a.d2 = __subc   (b.d2, tmp96.d2);

#ifdef DEBUG_GPU_MATH
  if(f.d2)						// check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
  {
    MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2, 13) // a should never have a value >= 2^80, if so square_96_160() will overflow!
  }							// this will warn whenever a becomes close to 2^80
#endif

#ifdef CPU_SIEVE
  shifter<<= 32 - shiftcount;
#endif
  while(shifter)
  {
							// On input a is at most 79.322 bits (see end of this loop)

    square_96_160(&b, a);				// b = a^2, b is at most 158.644 bits

    if(shifter&0x80000000)shl_192(&b);			// Optional multiply by 2.  At this point b can be 159.644 bits.

    a.d0 = b.d2;					// a = floor (b / 2^64)
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3_special(&tmp192, a, u);		// tmp192 = (b / 2^64) * (2 ^ 160 / f)     (ignore the floor functions for now)

    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// In the case we care about most (large f values that might cause b to exceed 160 bits),
						        // the quotient is off by at most 4.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2, whereas mul_96_192_no_low3_special adds only 2 partial results,
						        // which could have generated 3 more carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2.  Since we know the maximum value of b, the maximum value of a.d2
							// is 2^31.17.  Similarly, a was generated by a floor function, and could thus be
							// low by almost 1.  If we account for this a value up to u.d2 could have been added
							// into tmp192.d2.  Since we know the maximum value of f is 77 bits, the maximum value
							// of u is 160-77 (83) bits.  Thus the maximum value of u.d2 is 2^19.
							// Since maximum a.d2 + maximum u.d2 is less than 2^32, these 2 values combined can
							// only generate only 1 carry into tmp192.d3 -- for a total of up to 4 carries lost.

    mul_96(&tmp96, a, f);				// tmp96 = quotient * f, we only compute the low 96-bits here

    a.d0 = __sub_cc (b.d0, tmp96.d0);			// Compute the remainder
    a.d1 = __subc_cc(b.d1, tmp96.d1);			// we do not need the upper digits of b and tmp96 because the result is 0 after subtraction!
    a.d2 = __subc   (b.d2, tmp96.d2);
							// Since the quotient was up to 4 too small, the remainder has a maximum value of 5*f,
							// or 77 bits + log2 (5) bits, which is 79.322 bits.  In theory, this kernel can handle
							// f values up to 2^77.178.

#ifdef DEBUG_GPU_MATH
    if(f.d2)						// check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
    {
      MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2, 13) // a should never have a value >= 2^80, if so square_96_160() will overflow!
    }							// this will warn whenever a becomes close to 2^80
#endif

//    shifter<<=1;
    shifter += shifter;
  }
  
/*#ifndef DEBUG_GPU_MATH
  mod_simple_96(&a, tmp96, f, ff);			// Adjustment.  The code above may produce an a that is too large by up to 5 times f.
#else
  mod_simple_96(&a, tmp96, f, ff, 0, 79 - 64, 4, modbasecase_debug);
#endif*/

/* finally check if we found a factor and write the factor to RES[]
this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
produce an a that is too large by up to 5 times f. */
  mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}


__device__ static void test_FC96_barrett76(int96 f, int192 b, unsigned int shifter, unsigned int *RES
#ifdef CPU_SIEVE
                                           , int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           , int bit_max64, unsigned int *modbasecase_debug
#endif
                                           )
{
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  float ff;

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);	// f.d0 ignored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!
  ff=__int_as_float(0x3f7ffffb) / ff;			// just a little bit below 1.0f so we always underestimate the quotient

#ifndef DEBUG_GPU_MATH
  inv_160_96(&u,f,ff);					// u = floor(2^160 / f)
#else
  inv_160_96(&u,f,ff,modbasecase_debug);		// u = floor(2^160 / f)
#endif

  a.d0 = b.d2;						// a = floor(b / 2^64)
  a.d1 = b.d3;
  a.d2 = b.d4;

  mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2^64) * (2 ^ 160 / f)     (ignore the floor functions for now)

  a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);					// tmp96 = quotient * f, we only compute the low 96-bits here

  a.d0 = __sub_cc (b.d0, tmp96.d0);			// Compute the remainder
  a.d1 = __subc_cc(b.d1, tmp96.d1);			// we do not need the upper digits of b and tmp96 because the result is 0 after subtraction!
  a.d2 = __subc   (b.d2, tmp96.d2);

#ifdef DEBUG_GPU_MATH
  if(f.d2)						// check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
  {
    MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2, 13) // a should never have a value >= 2^80, if so square_96_160() will overflow!
  }							// this will warn whenever a becomes close to 2^80
#endif

#ifdef CPU_SIEVE
  shifter<<= 32 - shiftcount;
#endif
  while(shifter)
  {
							// On input a is at most 79.585 bits (see end of this loop)

    square_96_160(&b, a);				// b = a^2, b is at most 159.17 bits

    a.d0 = b.d2;					// a = floor (b / 2^64)
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2^64) * (2 ^ 160 / f)     (ignore the floor functions for now)

    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// In the case we care about most (large f values that might cause b to exceed 160 bits),
						        // the quotient is off by at most 5.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2 which could have generated 4 carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2.  Since we know the maximum value of b, the maximum value of a.d2
							// is 2^31.17.  Similarly, a was generated by a floor function, and could thus be
							// low by almost 1.  If we account for this a value up to u.d2 could have been added
							// into tmp192.d2.  Since we know the maximum value of f is 76 bits, the maximum value
							// of u is 160-76 (84) bits.  Thus the maximum value of u.d2 is 2^20.
							// Since maximum a.d2 + maximum u.d2 is less than 2^32, these 2 values combined can
							// only generate only 1 carry into tmp192.d3 -- for a total of up to 5 carries lost.

    mul_96(&tmp96, a, f);				// tmp96 = quotient * f, we only compute the low 96-bits here

    a.d0 = __sub_cc (b.d0, tmp96.d0);			// Compute the remainder
    a.d1 = __subc_cc(b.d1, tmp96.d1);			// we do not need the upper digits of b and tmp96 because the result is 0 after subtraction!
    a.d2 = __subc   (b.d2, tmp96.d2);
							// Since the quotient was up to 5 too small, the remainder has a maximum value of 6*f,
							// or 76 bits + log2 (6) bits, which is 78.585 bits.  In theory, this kernel can handle
							// f values up to 2^76.415.

    if(shifter&0x80000000)shl_96(&a);			// "optional multiply by 2" as in Prime95 documentation
							// At this point a can be 79.585 bits.

#ifdef DEBUG_GPU_MATH
    if(f.d2)						// check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
    {
      MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2, 13) // a should never have a value >= 2^80, if so square_96_160() will overflow!
    }							// this will warn whenever a becomes close to 2^80
#endif

//    shifter<<=1;
    shifter += shifter;
  }

/*#ifndef DEBUG_GPU_MATH
  mod_simple_96(&a, tmp96, f, ff);			// Adjustment.  The code above may produce an a that is too large by up to 11 times f.
#else
  mod_simple_96(&a, tmp96, f, ff, 0, 79 - 64, 11, modbasecase_debug);
#endif*/

/* finally check if we found a factor and write the factor to RES[]
this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
produce an a that is too large by up to 11 times f. */
  mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}
