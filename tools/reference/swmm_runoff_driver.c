/* Component adapter for untouched EPA SWMM 5.2.4 runoff source.
 * No full runoff driver clocks, weather cursors, or adaptive timestep selection.
 * Include original process sources and link every remaining original solver TU.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(SWMM_SUBCATCH_SOURCE) || !defined(SWMM_INFIL_SOURCE) || !defined(SWMM_ODE_HEADER)
#error "Define absolute SWMM_SUBCATCH_SOURCE, SWMM_INFIL_SOURCE, SWMM_ODE_HEADER paths"
#endif
#include SWMM_ODE_HEADER
int runoff_observed_integrate(double[], int, double, double, double, double,
    void (*)(double, double*, double*));
#define odesolve_integrate runoff_observed_integrate
#include SWMM_SUBCATCH_SOURCE
#undef odesolve_integrate

/* Only the infiltration parameter bridge uses native feet/seconds directly.
 * Full-engine UCF, including the original rain-gage bridge, remains unchanged.
 */
double runoff_native_units(int quantity);
#define UCF runoff_native_units
#include SWMM_INFIL_SOURCE
#undef UCF

double runoff_native_units(int quantity)
{
    (void)quantity;
    return 1.0;
}

static int OdeCalls[3];
static int OdeError[3];
static double Clock;
static double FracZero;
static int RouteCode;
static double RoutedFraction;
static int Initialized;

typedef struct {
    double dt, requested_rain, actual_rain, evap, external;
    TSubarea before[3];
    double routed_inflow[3];
    double storage_before, storage_after, rain_volume, external_volume;
    double internal_volume, total_inflow_volume, native_runon, endpoint_all, storage_outflow, residual_all;
    double outlet_storage_outflow, residual_outlet, routing_lag_residual, runoff_return;
    int probe_kind, probe_subarea;
    double probe_time, probe_rate, probe_depth, probe_tolerance, probe_initial_step;
} TObservation;

static TObservation Observation;

/* Record the original return code without changing arguments, state, or result. */
int runoff_observed_integrate(double state[], int n, double x1, double x2,
    double eps, double h1, void (*derivs)(double, double*, double*))
{
    int code = odesolve_integrate(state, n, x1, x2, eps, h1, derivs);
    int i;
    for (i = 0; i < 3; ++i) if (theSubarea == &Subcatch[0].subArea[i]) {
        ++OdeCalls[i];
        if (code != 0) OdeError[i] = code;
    }
    return code;
}

#include "swmm_runoff_output.h"

static int command_error(size_t line, const char *operation, const char *reason)
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

static int allocate_engine(void)
{
    UnitSystem = US;
    FlowUnits = CFS;
    IgnoreSnowmelt = TRUE;
    IgnoreGwater = TRUE;
    IgnoreQuality = TRUE;
    IgnoreRainfall = FALSE;
    InfilModel = MOD_GREEN_AMPT;
    Adjust.hydconFactor = 1.0;
    Evap.recoveryFactor = 1.0;
    RouteStep = 1.0;
    MinRouteStep = 1.0;
    Frpt.file = stderr;
    Nobjects[SUBCATCH] = 1;
    Nobjects[GAGE] = 1;
    Nobjects[NODE] = 1;
    Subcatch = calloc(1, sizeof(TSubcatch));
    Gage = calloc(1, sizeof(TGage));
    Node = calloc(1, sizeof(TNode));
    if (!Subcatch || !Gage || !Node) return 0;
    infil_create(1);
    lid_create(0, 1);
    if (ErrorCode != 0 || !odesolve_open(1)) return 0;
    return 1;
}

static void clear_observation(void)
{
    memset(&Observation, 0, sizeof(Observation));
    Observation.probe_subarea = -1;
    memset(OdeCalls, 0, sizeof(OdeCalls));
    memset(OdeError, 0, sizeof(OdeError));
    Vevap = Vpevap = Vinfil = Vinflow = Voutflow = 0.0;
    VlidIn = VlidInfil = VlidOut = VlidDrain = VlidReturn = 0.0;
}

static int initialize(const double *v)
{
    int i;
    TSubcatch *s = &Subcatch[0];
    if (v[0] <= 0.0 || v[1] < 0.0 || v[1] > 1.0 ||
        v[2] < 0.0 || v[2] > 1.0 || v[3] < 0.0 || v[4] < 0.0 ||
        v[5] < 0.0 || v[6] < 0.0 || v[7] < 0.0 || v[8] < 0.0 ||
        (v[9] != TO_OUTLET && v[9] != TO_IMPERV && v[9] != TO_PERV) ||
        v[10] < 0.0 || v[10] > 1.0 || (v[14] != 0.0 && v[14] != 1.0)) return 0;
    if (v[9] != TO_OUTLET && (v[1] <= 0.0 || v[1] >= 1.0)) return 0;
    if (v[9] == TO_IMPERV && v[10] > 0.0 && v[2] == 1.0) return 0;
    {
        double parameters[3] = {v[11], v[12], v[13]};
        if (!grnampt_setParams(&Infil[0].grnAmpt, parameters)) return 0;
    }
    s->ID = "native-runoff-component";
    s->gage = 0;
    s->outNode = 0;                 /* External outlet, no hydraulic routing. */
    s->outSubcatch = -1;
    s->infil = 0;
    s->infilModel = MOD_GREEN_AMPT;
    s->nPervPattern = s->dStorePattern = s->infilPattern = -1;
    s->area = v[0];
    s->fracImperv = v[1];
    s->width = v[3];
    s->slope = v[4];
    FracZero = v[2];
    RouteCode = (int)v[9];
    RoutedFraction = v[10];
    Evap.dryOnly = (int)v[14];
    s->subArea[IMPERV0].fArea = v[1] * v[2];
    s->subArea[IMPERV1].fArea = v[1] * (1.0 - v[2]);
    s->subArea[PERV].fArea = 1.0 - v[1];
    s->subArea[IMPERV0].N = s->subArea[IMPERV1].N = v[5];
    s->subArea[PERV].N = v[6];
    s->subArea[IMPERV0].dStore = 0.0;
    s->subArea[IMPERV1].dStore = v[7];
    s->subArea[PERV].dStore = v[8];
    for (i = 0; i < 3; ++i) {
        s->subArea[i].routeTo = TO_OUTLET;
        s->subArea[i].fOutlet = 1.0;
    }
    if (RouteCode == TO_IMPERV) {
        s->subArea[PERV].routeTo = TO_IMPERV;
        s->subArea[PERV].fOutlet = 1.0 - RoutedFraction;
    }
    if (RouteCode == TO_PERV) for (i = IMPERV0; i <= IMPERV1; ++i) {
        s->subArea[i].routeTo = TO_PERV;
        s->subArea[i].fOutlet = 1.0 - RoutedFraction;
    }
    subcatch_validate(0);            /* Original combined-impervious alpha. */
    subcatch_initState(0);
    if (Initialized) stats_close();
    if (stats_open() != 0 || ErrorCode != 0) return 0;
    Clock = OldRunoffTime = NewRunoffTime = 0.0;
    Initialized = 1;
    clear_observation();
    Observation.storage_before = Observation.storage_after = subcatch_getStorage(0);
    return 1;
}

static int restore_state(const double *v)
{
    int i;
    double soil[5];
    TSubcatch *s = &Subcatch[0];
    TGrnAmpt *g = &Infil[0].grnAmpt;
    double capacity = g->IMDmax * g->Lu;
    double ceiling = fmax(g->IMDmax, capacity / g->Lu);
    if (v[0] < 0.0 || v[12] != 0.0 || v[16] < 0.0 || v[16] > ceiling ||
        v[18] < 0.0 || v[18] > capacity || (v[19] != 0.0 && v[19] != 1.0)) return 0;
    for (i = 0; i < 3; ++i) if (v[1 + i*3] < 0.0 || v[3 + i*3] < 0.0) return 0;
    for (i = 10; i <= 15; ++i) if (v[i] < 0.0) return 0;
    Clock = v[0];
    OldRunoffTime = NewRunoffTime = Clock * 1000.0;
    if (!isfinite(NewRunoffTime)) return 0;
    for (i = 0; i < 3; ++i) {
        s->subArea[i].depth = v[1 + i*3];
        s->subArea[i].inflow = v[2 + i*3];
        s->subArea[i].runoff = v[3 + i*3];
    }
    s->oldRunoff = v[10]; s->newRunoff = v[11]; s->runon = v[12];
    s->rainfall = v[13]; s->evapLoss = v[14]; s->infilLoss = v[15];
    for (i = 0; i < 5; ++i) soil[i] = v[16 + i];
    grnampt_setState(g, soil);
    clear_observation();
    Observation.storage_before = Observation.storage_after = subcatch_getStorage(0);
    return 1;
}

static int advance(const double *v)
{
    int i;
    TSubcatch *s = &Subcatch[0];
    double supplied, all_inflow = 0.0;
    if (v[0] <= 0.0 || v[1] < 0.0 || v[2] < 0.0 || v[3] < 0.0) return 0;
    supplied = v[1] * UCF(RAINFALL);
    if (!isfinite(supplied) || (v[1] > 0.0 && supplied == 0.0) ||
        !isfinite(Clock + v[0]) || Clock + v[0] <= Clock ||
        !isfinite((Clock + v[0]) * 1000.0)) return 0;
    clear_observation();
    Observation.dt = v[0]; Observation.requested_rain = v[1];
    Observation.evap = v[2]; Observation.external = v[3];
    Observation.storage_before = subcatch_getStorage(0);
    for (i = 0; i < 3; ++i) Observation.before[i] = s->subArea[i];
    OldRunoffTime = Clock * 1000.0;
    Clock += v[0];
    NewRunoffTime = Clock * 1000.0;
    Gage[0].rainfall = supplied;
    Evap.rate = v[2];
    subcatch_setOldState(0);
    subcatch_getRunon(0);
    for (i = 0; i < 3; ++i) {
        double area = s->area * s->subArea[i].fArea;
        Observation.internal_volume += s->subArea[i].inflow * area * v[0];
    }
    subcatch_addRunonFlow(0, v[3]);
    Observation.native_runon = s->runon;
    for (i = 0; i < 3; ++i) {
        Observation.routed_inflow[i] = s->subArea[i].inflow;
        double area = s->area * s->subArea[i].fArea;
        all_inflow += s->subArea[i].inflow * area * v[0];
    }
    Observation.runoff_return = subcatch_getRunoff(0, v[0]);
    Observation.actual_rain = s->rainfall;
    Observation.storage_after = subcatch_getStorage(0);
    Observation.rain_volume = s->rainfall * v[0] * s->area;
    Observation.external_volume = s->runon * v[0] * s->area;
    Observation.total_inflow_volume = all_inflow;
    for (i = 0; i < 3; ++i) {
        double area = s->area * s->subArea[i].fArea;
        Observation.endpoint_all += s->subArea[i].runoff * area * v[0];
    }
    Observation.storage_outflow = Observation.storage_before + Observation.rain_volume + all_inflow
        - Vevap - Vinfil - Observation.storage_after;
    Observation.residual_all = Observation.endpoint_all - Observation.storage_outflow;
    Observation.outlet_storage_outflow = Observation.storage_before + Observation.rain_volume
        + Observation.external_volume - Vevap - Vinfil - Observation.storage_after;
    Observation.residual_outlet = Voutflow - Observation.outlet_storage_outflow;
    Observation.routing_lag_residual = Observation.internal_volume - (Observation.endpoint_all - Voutflow);
    s->runon = 0.0;                 /* Original runoff_execute accepted endpoint. */
    return 1;
}

static int probe_rate(const double *v)
{
    int i;
    double saved_alpha = Alpha, saved_store = Dstore;
    TSubarea copy;
    if (v[0] < 0.0 || v[0] > 2.0 || v[0] != floor(v[0]) || v[1] <= 0.0) return 0;
    i = (int)v[0];
    copy = Subcatch[0].subArea[i];
    Alpha = copy.alpha;
    Dstore = copy.dStore;
    clear_observation();
    Observation.probe_kind = 1;
    Observation.probe_subarea = i;
    Observation.probe_time = v[1];
    Observation.probe_rate = findSubareaRunoff(&copy, v[1]);
    Observation.probe_depth = copy.depth;
    Observation.storage_before = Observation.storage_after = subcatch_getStorage(0);
    Alpha = saved_alpha;
    Dstore = saved_store;
    return 1;
}

/* Controller observation uses the original derivative and a copied native state. */
static int probe_ode(const double *v)
{
    int i;
    double saved_alpha = Alpha, saved_store = Dstore;
    TSubarea *saved_subarea = theSubarea;
    TSubarea copy;
    if (v[0] < 0.0 || v[0] > 2.0 || v[0] != floor(v[0]) || v[1] <= 0.0 || v[2] <= 0.0 || v[3] <= 0.0) return 0;
    i = (int)v[0];
    copy = Subcatch[0].subArea[i];
    Alpha = copy.alpha;
    Dstore = copy.dStore;
    theSubarea = &copy;
    clear_observation();
    Observation.probe_kind = 2;
    Observation.probe_subarea = i;
    Observation.probe_time = v[1];
    Observation.probe_tolerance = v[2];
    Observation.probe_initial_step = v[3];
    OdeCalls[i] = 1;
    OdeError[i] = odesolve_integrate(&copy.depth, 1, 0.0, v[1], v[2], v[3], getDdDt);
    Observation.probe_depth = copy.depth;
    Observation.storage_before = Observation.storage_after = subcatch_getStorage(0);
    Alpha = saved_alpha;
    Dstore = saved_store;
    theSubarea = saved_subarea;
    return 1;
}

int main(void)
{
    char line[4096];
    size_t index = 0;
    int result = 0;
    if (!allocate_engine()) return command_error(0, "allocate", "native allocation error: provide memory and retry");
    emit_header();
    while (fgets(line, sizeof(line), stdin)) {
        char *operation, *token;
        double values[21];
        size_t count = 0;
        ++index;
        if (!strchr(line, '\n') && !feof(stdin)) { result = command_error(index, "input", "line exceeds 4095 bytes: shorten the command"); break; }
        operation = strtok(line, " \t\r\n");
        if (!operation) { result = command_error(index, "input", "empty command: use INIT, STATE, STEP, or RATE"); break; }
        while ((token = strtok(NULL, " \t\r\n"))) {
            if (count == 21 || !number(token, &values[count])) break;
            ++count;
        }
        if (token) { result = command_error(index, operation, "invalid or excess numeric fields: supply finite round-tripping numbers"); break; }
        if (!strcmp(operation, "INIT")) {
            if (count != 15 || !initialize(values)) { result = command_error(index, operation, "require admitted 15-field native config; exclude unsupported processes and invalid routing"); break; }
        } else if (!strcmp(operation, "STATE")) {
            if (!Initialized || count != 21 || !restore_state(values)) { result = command_error(index, operation, "send INIT then 21 state fields; accepted runon must be zero and soil/depth/rates admissible"); break; }
        } else if (!strcmp(operation, "STEP")) {
            if (!Initialized || count != 4 || !advance(values)) { result = command_error(index, operation, "send INIT then positive dt and nonnegative native rain/evap/external-runon; require representable clock/bridge"); break; }
        } else if (!strcmp(operation, "RATE")) {
            if (!Initialized || count != 2 || !probe_rate(values)) { result = command_error(index, operation, "send INIT then RATE subarea_index positive_tRunoff with index 0..2"); break; }
        } else if (!strcmp(operation, "ODE")) {
            if (!Initialized || count != 4 || !probe_ode(values)) { result = command_error(index, operation, "send INIT then ODE subarea_index positive_dt positive_tolerance positive_initial_step"); break; }
        } else { result = command_error(index, operation, "unknown command: use INIT, STATE, STEP, RATE, or ODE"); break; }
        if (!emit_row(operation, index)) { result = command_error(index, operation, "native output is nonfinite: retain the input as unsupported numerical evidence"); break; }
    }
    if (ferror(stdin)) result = command_error(index, "input", "stdin read error: supply readable input");
    if (fflush(stdout)) result = command_error(index, "output", "stdout write error: supply writable output");
    if (Initialized) stats_close();
    odesolve_close();
    lid_delete();
    infil_delete();
    free(Subcatch); free(Gage); free(Node);
    return result;
}
