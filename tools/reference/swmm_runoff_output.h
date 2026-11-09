/* CSV observations of the admitted native component and original source codes. */
#ifndef HYDR_SWMM_RUNOFF_OUTPUT_H
#define HYDR_SWMM_RUNOFF_OUTPUT_H

static void emit_header(void)
{
    int i;
    fputs("op,index,time_s,dt_s,requested_rain_ft_s,actual_rain_ft_s,evap_ft_s,external_runon_cfs"
        ",area_ft2,frac_imperv,frac_zero_imperv,width_ft,slope,n_imperv,n_perv"
        ",dstore_imperv_ft,dstore_perv_ft,route_code,routed_fraction,evap_dry_only"
        ",S,Ks,IMDmax,Lu,IMD,F,Fu,Sat,T,oldRunoff,newRunoff,runon,rainfall,evapLoss,infilLoss", stdout);
    for (i = 0; i < 3; ++i) printf(",a%d_fArea,a%d_N,a%d_dStore,a%d_alpha,a%d_routeTo,a%d_fOutlet"
        ",a%d_depth,a%d_inflow,a%d_runoff,a%d_old_depth,a%d_old_inflow,a%d_old_runoff,a%d_routed_inflow",
        i,i,i,i,i,i,i,i,i,i,i,i,i);
    fputs(",Vinflow,Vevap,Vpevap,Vinfil,Voutflow,storage_before_ft3,storage_after_ft3,rain_volume_ft3"
        ",external_volume_ft3,internal_routing_volume_ft3,total_subarea_inflow_volume_ft3,native_runon_ft_s,endpoint_all_volume_ft3,endpoint_outlet_volume_ft3"
        ",storage_implied_outflow_ft3,outlet_storage_implied_outflow_ft3,storage_residual_all_ft3,storage_residual_outlet_ft3,internal_routing_lag_residual_ft3,runoff_return_ft_s"
        ",ode_calls0,ode_error0,ode_calls1,ode_error1,ode_calls2,ode_error2,native_error_code"
        ",probe_kind,probe_subarea,probe_time_s,probe_rate_ft_s,probe_depth_after_ft,probe_tolerance,probe_initial_step_s\n", stdout);
}

static int field(double value)
{
    printf(",%.17g", value);
    return isfinite(value);
}

static int emit_row(const char *operation, size_t index)
{
    int i, finite = 1;
    TSubcatch *s = &Subcatch[0];
    TGrnAmpt *g = &Infil[0].grnAmpt;
    printf("%s,%zu", operation, index);
#define OUT(value) do { finite &= field(value); } while (0)
    OUT(Clock); OUT(Observation.dt); OUT(Observation.requested_rain);
    OUT(Observation.actual_rain); OUT(Observation.evap); OUT(Observation.external);
    OUT(s->area); OUT(s->fracImperv); OUT(FracZero); OUT(s->width); OUT(s->slope);
    OUT(s->subArea[IMPERV0].N); OUT(s->subArea[PERV].N);
    OUT(s->subArea[IMPERV1].dStore); OUT(s->subArea[PERV].dStore);
    OUT(RouteCode); OUT(RoutedFraction); OUT(Evap.dryOnly);
    OUT(g->S); OUT(g->Ks); OUT(g->IMDmax); OUT(g->Lu);
    OUT(g->IMD); OUT(g->F); OUT(g->Fu); OUT(g->Sat); OUT(g->T);
    OUT(s->oldRunoff); OUT(s->newRunoff); OUT(s->runon);
    OUT(s->rainfall); OUT(s->evapLoss); OUT(s->infilLoss);
    for (i = 0; i < 3; ++i) {
        TSubarea *a = &s->subArea[i], *old = &Observation.before[i];
        OUT(a->fArea); OUT(a->N); OUT(a->dStore); OUT(a->alpha);
        OUT(a->routeTo); OUT(a->fOutlet); OUT(a->depth); OUT(a->inflow); OUT(a->runoff);
        OUT(old->depth); OUT(old->inflow); OUT(old->runoff); OUT(Observation.routed_inflow[i]);
    }
    OUT(Vinflow); OUT(Vevap); OUT(Vpevap); OUT(Vinfil); OUT(Voutflow);
    OUT(Observation.storage_before); OUT(Observation.storage_after); OUT(Observation.rain_volume);
    OUT(Observation.external_volume); OUT(Observation.internal_volume);
    OUT(Observation.total_inflow_volume); OUT(Observation.native_runon);
    OUT(Observation.endpoint_all); OUT(Voutflow); OUT(Observation.storage_outflow);
    OUT(Observation.outlet_storage_outflow); OUT(Observation.residual_all);
    OUT(Observation.residual_outlet); OUT(Observation.routing_lag_residual); OUT(Observation.runoff_return);
    for (i = 0; i < 3; ++i) { OUT(OdeCalls[i]); OUT(OdeError[i]); }
    OUT(ErrorCode); OUT(Observation.probe_kind); OUT(Observation.probe_subarea);
    OUT(Observation.probe_time); OUT(Observation.probe_rate); OUT(Observation.probe_depth);
    OUT(Observation.probe_tolerance); OUT(Observation.probe_initial_step);
#undef OUT
    putchar('\n');
    return finite;
}
#endif
