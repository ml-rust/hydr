/* Pinned EPA SWMM v5.2.4 drainage worker with complete hidden-state snapshot and restore.
 *
 * Includes untouched SWMM v5.2.4 routing, dynamic-wave, controls, mass-balance, and top-level
 * translation units so every static and heap-allocated state variable is captured and restored
 * bit-for-bit without modifying any original SWMM source file.
 */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(SWMM_SWMM5_SOURCE) || !defined(SWMM_ROUTING_SOURCE) \
    || !defined(SWMM_DYNWAVE_SOURCE) || !defined(SWMM_CONTROLS_SOURCE) \
    || !defined(SWMM_MASSBAL_SOURCE)
#error "Define SWMM_SWMM5_SOURCE, SWMM_ROUTING_SOURCE, SWMM_DYNWAVE_SOURCE, SWMM_CONTROLS_SOURCE, SWMM_MASSBAL_SOURCE"
#endif

#include SWMM_SWMM5_SOURCE
#include SWMM_ROUTING_SOURCE
#include SWMM_DYNWAVE_SOURCE
#include SWMM_CONTROLS_SOURCE
#include SWMM_MASSBAL_SOURCE

#include "swmm_drainage_snapshot.h"

static uint64_t dbl_bits(double x)
{
    uint64_t u = 0;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static int parse_hex_dbl(const char *s, double *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long u = strtoull(s, &end, 16);
    if (end == s || *end != '\0' || errno != 0) return 0;
    uint64_t bits = (uint64_t)u;
    memcpy(out, &bits, sizeof(*out));
    return isfinite(*out);
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static void print_state(void)
{
    double final_stor = massbal_getStorage(TRUE);
    double total_in = FlowTotals.dwInflow + FlowTotals.wwInflow + FlowTotals.gwInflow
        + FlowTotals.iiInflow + FlowTotals.exInflow + FlowTotals.initStorage;
    double total_out = FlowTotals.flooding + FlowTotals.outflow + FlowTotals.evapLoss
        + FlowTotals.seepLoss + final_stor;
    double pct_err = (total_in > 0.0) ? (100.0 * (1.0 - total_out / total_in)) : 0.0;
    printf("STATE %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
           " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
           dbl_bits(NewRoutingTime / 1000.0),
           dbl_bits(VariableStep),
           dbl_bits(FlowTotals.initStorage),
           dbl_bits(FlowTotals.exInflow),
           dbl_bits(FlowTotals.outflow),
           dbl_bits(FlowTotals.flooding),
           dbl_bits(FlowTotals.evapLoss),
           dbl_bits(FlowTotals.seepLoss),
           dbl_bits(final_stor),
           dbl_bits(pct_err));
    for (int i = 0; i < Nobjects[NODE]; ++i) {
        TNode *n = &Node[i];
        printf("NODE %d %s %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
               " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
               i, n->ID,
               dbl_bits(n->invertElev),
               dbl_bits(n->fullDepth),
               dbl_bits(n->newDepth),
               dbl_bits(n->newVolume),
               dbl_bits(n->inflow),
               dbl_bits(n->outflow),
               dbl_bits(n->overflow));
    }
    for (int i = 0; i < Nobjects[LINK]; ++i) {
        TLink *l = &Link[i];
        printf("LINK %d %s %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
               " %016" PRIx64 " %016" PRIx64 "\n",
               i, l->ID,
               dbl_bits(l->newFlow),
               dbl_bits(l->newDepth),
               dbl_bits(l->newVolume),
               dbl_bits(l->setting),
               dbl_bits(l->targetSetting),
               dbl_bits(l->timeLastSet));
    }
    printf("END_STATE\n");
    fflush(stdout);
}

int main(void)
{
    static char line[1048576];
    int opened = 0;
    while (fgets(line, (int)sizeof(line), stdin) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        if (strncmp(line, "OPEN ", 5) == 0) {
            if (opened) { printf("ERR already opened\n"); fflush(stdout); continue; }
            char *inp = line + 5;
            char *sp = strchr(inp, ' ');
            if (!sp) { printf("ERR OPEN requires inp and rpt paths\n"); fflush(stdout); continue; }
            *sp = '\0';
            char *rpt = sp + 1;
            int err = swmm_open(inp, rpt, "");
            if (err) {
                printf("ERR swmm_open=%d %s\n", err, ErrorMsg);
                fflush(stdout);
                continue;
            }
            err = swmm_start(FALSE);
            if (err) {
                swmm_close();
                printf("ERR swmm_start=%d %s\n", err, ErrorMsg);
                fflush(stdout);
                continue;
            }
            opened = 1;
            printf("OK %d %d %d %d %d %d %d %d %d %d\n",
                   Nobjects[NODE], Nobjects[LINK], Nlinks[CONDUIT], Nnodes[STORAGE],
                   Nnodes[OUTFALL], Nlinks[ORIFICE], Nlinks[WEIR], RuleCount,
                   SurchargeMethod, UnitSystem);
            fflush(stdout);
        } else if (strncmp(line, "SET_INFLOW ", 11) == 0) {
            int idx = -1;
            char hex[32];
            double rate = 0.0;
            if (!opened || sscanf(line + 11, "%d %31s", &idx, hex) != 2
                || idx < 0 || idx >= Nobjects[NODE] || !parse_hex_dbl(hex, &rate)) {
                printf("ERR invalid SET_INFLOW\n");
            } else {
                Node[idx].apiExtInflow = rate;
                printf("OK\n");
            }
            fflush(stdout);
        } else if (strncmp(line, "STEP ", 5) == 0) {
            double dt_s = 0.0;
            if (!opened || !parse_hex_dbl(line + 5, &dt_s) || dt_s <= 0.0) {
                printf("ERR invalid STEP\n");
                fflush(stdout);
                continue;
            }
            double realRouteStep = RouteStep;
            double targetMs = NewRoutingTime + 1000.0 * dt_s;
            if (targetMs > TotalDuration) targetMs = TotalDuration;
            RoutingDuration = targetMs;
            if (dt_s < RouteStep) RouteStep = dt_s;
            while (NewRoutingTime < RoutingDuration && !ErrorCode) {
                execRouting();
            }
            RouteStep = realRouteStep;
            RoutingDuration = TotalDuration;
            ElapsedTime = NewRoutingTime / MSECperDAY;
            if (ErrorCode) {
                printf("ERR step=%d %s\n", ErrorCode, ErrorMsg);
            } else {
                printf("OK %016" PRIx64 "\n", dbl_bits(NewRoutingTime / 1000.0));
            }
            fflush(stdout);
        } else if (strcmp(line, "STATE") == 0) {
            if (!opened) { printf("ERR not opened\n"); fflush(stdout); continue; }
            print_state();
        } else if (strcmp(line, "SNAPSHOT") == 0) {
            if (!opened) { printf("ERR not opened\n"); fflush(stdout); continue; }
            Buf b;
            buf_init(&b);
            if (!capture_snapshot(&b)) {
                buf_free(&b);
                printf("ERR snapshot failed\n");
            } else {
                printf("SNAPSHOT ");
                for (size_t i = 0; i < b.len; ++i) printf("%02x", b.data[i]);
                printf("\n");
                buf_free(&b);
            }
            fflush(stdout);
        } else if (strncmp(line, "RESTORE ", 8) == 0) {
            if (!opened) { printf("ERR not opened\n"); fflush(stdout); continue; }
            const char *hex = line + 8;
            size_t hlen = strlen(hex);
            if (hlen == 0 || (hlen % 2) != 0) {
                printf("ERR invalid snapshot hex length\n");
                fflush(stdout);
                continue;
            }
            Buf b;
            buf_init(&b);
            b.data = (unsigned char *)malloc(hlen / 2);
            if (!b.data) { printf("ERR oom\n"); fflush(stdout); continue; }
            b.len = b.cap = hlen / 2;
            int bad = 0;
            for (size_t i = 0; i < b.len; ++i) {
                int hi = hex_nibble(hex[2 * i]);
                int lo = hex_nibble(hex[2 * i + 1]);
                if (hi < 0 || lo < 0) { bad = 1; break; }
                b.data[i] = (unsigned char)((hi << 4) | lo);
            }
            if (bad || !restore_snapshot(&b)) {
                printf("ERR restore failed\n");
            } else {
                printf("OK\n");
            }
            buf_free(&b);
            fflush(stdout);
        } else if (strcmp(line, "CLOSE") == 0) {
            if (opened) {
                swmm_end();
                swmm_close();
                opened = 0;
            }
            printf("OK\n");
            fflush(stdout);
            break;
        } else {
            printf("ERR unknown command\n");
            fflush(stdout);
        }
    }
    if (opened) {
        swmm_end();
        swmm_close();
    }
    return 0;
}
