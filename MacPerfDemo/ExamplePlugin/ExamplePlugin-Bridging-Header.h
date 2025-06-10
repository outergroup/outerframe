//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static inline int toPrecisionThriftyC(float d, int precision, char* output, size_t outputSize) {
    if (d >= 1.0 && (log10(d) + 1) >= (float)precision) {
        return snprintf(output, outputSize, "%.0f", d);
    }

    int preciseLen = snprintf(output, outputSize, "%.*g", precision, d);
    double parsedPrecise = strtod(output, NULL);

    // Consider returning shorter output
    char candidate[64];
    for (int i = 0; i < precision; i++) {
        int len = snprintf(candidate, sizeof(candidate), "%.*g", i, d);
        if (len >= 0 && strtod(candidate, NULL) == parsedPrecise) {
            strncpy(output, candidate, outputSize);
            return len;
        }
    }

    // Use standard snprintf output
    return preciseLen;
}

// Direct buffer formatting for scientific notation
static inline int formatScientificC(float d, char* output, size_t outputSize) {
    if (outputSize == 0) return 0;
    int len = snprintf(output, outputSize, "%.1e", d);
    return (len >= 0 && len < outputSize) ? len : 0;
}
