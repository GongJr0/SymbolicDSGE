#ifndef SDSGE_TIME_H
#define SDSGE_TIME_H

#include "../_common/sdsge_common.h"

/* sdsge_monotime_sec: monotonic seconds from an unspecified origin, where only
 * the difference between two reads carries meaning. No initialization and no
 * shared state; callable from any thread. */

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

static inline f64 sdsge_monotime_sec(void) {
  LARGE_INTEGER freq, t;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&t);
  return (f64)t.QuadPart / (f64)freq.QuadPart;
}
#else
#include <time.h>

static inline f64 sdsge_monotime_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (f64)ts.tv_sec + (f64)ts.tv_nsec * 1e-9;
}
#endif
#endif // !SDSGE_TIME_H
