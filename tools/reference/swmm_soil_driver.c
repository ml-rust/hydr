/* Native-input adapter for the untouched pinned EPA infiltration source. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SWMM_SOIL_SOURCE
#error "Define SWMM_SOIL_SOURCE as the absolute pinned infil.c path"
#endif
#include SWMM_SOIL_SOURCE

TEvap Evap;

/* Parameters already use native feet and seconds. */
double UCF(int quantity)
{
    (void)quantity;
    return 1.0;
}

static int error(size_t line, const char *operation, const char *reason)
{
    fprintf(stderr, "line %zu command %s: %s\n", line, operation, reason);
    return 2;
}

static int number(const char *token, double *value)
{
    char *end = NULL;
    errno = 0;
    *value = strtod(token, &end);
    return end != token && *end == '\0' && isfinite(*value) &&
        (errno == 0 || (errno == ERANGE && *value != 0.0));
}

static int finite_state(const TGrnAmpt *soil)
{
    return isfinite(soil->S) && isfinite(soil->Ks) && isfinite(soil->IMDmax) &&
        isfinite(soil->Lu) && isfinite(soil->IMD) && isfinite(soil->F) &&
        isfinite(soil->Fu) && isfinite(soil->T);
}

static void emit(const char *operation, size_t index, const TGrnAmpt *soil,
    int has_rate, double rate, int has_root, double root)
{
    printf("%s,%zu,", operation, index);
    if (has_rate) printf("%.17g", rate);
    putchar(',');
    if (has_root) printf("%.17g", root);
    printf(",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n",
        soil->S, soil->Ks, soil->IMDmax, soil->Lu, soil->IMD, soil->F,
        soil->Fu, (int)soil->Sat, soil->T);
}

int main(void)
{
    TGrnAmpt soil = {0};
    char line[2048];
    size_t index = 0;
    int initialized = 0;
    InfilFactor = 1.0;
    Evap.recoveryFactor = 1.0;
    puts("op,index,rate,F2,S,Ks,IMDmax,Lu,IMD,F,Fu,Sat,T");
    while (fgets(line, sizeof(line), stdin) != NULL) {
        char *operation;
        char *token;
        double values[8];
        double rate = 0.0;
        double root = 0.0;
        size_t count = 0;
        int has_rate = 0;
        int has_root = 0;
        ++index;
        if (strchr(line, '\n') == NULL && !feof(stdin))
            return error(index, "input", "line exceeds 2047 bytes: shorten the command");
        operation = strtok(line, " \t\r\n");
        if (operation == NULL)
            return error(index, "input", "empty command: use INIT, STATE, STEP, or F2");
        while ((token = strtok(NULL, " \t\r\n")) != NULL) {
            if (count == 8)
                return error(index, operation, "too many fields: use the documented command shape");
            if (!number(token, &values[count]))
                return error(index, operation, "invalid numeric field: supply finite round-tripping numbers");
            ++count;
        }
        if (strcmp(operation, "INIT") == 0 || strcmp(operation, "STATE") == 0) {
            int restore = strcmp(operation, "STATE") == 0;
            if (count != (restore ? 8u : 3u))
                return error(index, operation, "require S Ks IMDmax followed by IMD F Fu Sat T for STATE");
            if (!grnampt_setParams(&soil, values))
                return error(index, operation, "native parameter check rejects S/Ks/IMDmax: require S>=0, Ks>0, and IMDmax in [0,1]");
            if (restore) {
                if (values[6] != 0.0 && values[6] != 1.0)
                    return error(index, operation, "Sat must be exactly zero or one");
                grnampt_setState(&soil, values + 3);
            }
            else grnampt_initState(&soil);
            initialized = 1;
        }
        else if (strcmp(operation, "STEP") == 0) {
            if (!initialized)
                return error(index, operation, "missing initial state: send INIT or STATE first");
            if (count != 4 || (values[0] != GREEN_AMPT && values[0] != MOD_GREEN_AMPT))
                return error(index, operation, "require model dt irate depth with model 2 or 3");
            if (values[1] <= 0.0 || values[3] < 0.0)
                return error(index, operation, "require positive dt and nonnegative ponded depth");
            rate = grnampt_getInfil(&soil, values[1], values[2], values[3], (int)values[0]);
            has_rate = 1;
        }
        else if (strcmp(operation, "F2") == 0) {
            if (!initialized || count != 4)
                return error(index, operation, "send INIT or STATE, then F2 f1 c1 ks ts");
            root = grnampt_getF2(values[0], values[1], values[2], values[3]);
            has_root = 1;
        }
        else return error(index, operation, "unknown command: use INIT, STATE, STEP, or F2");
        if (!finite_state(&soil) || (has_rate && !isfinite(rate)) || (has_root && !isfinite(root)))
            return error(index, operation, "native output is nonfinite: retain this input as an unsupported numerical case");
        emit(operation, index, &soil, has_rate, rate, has_root, root);
    }
    if (ferror(stdin)) return error(index, "input", "stdin read error: retry with a readable command stream");
    if (fflush(stdout) != 0) return error(index, "output", "stdout write error: supply writable output storage");
    return 0;
}
