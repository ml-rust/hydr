/* Complete hidden-state snapshot and restore for pinned EPA SWMM v5.2.4 drainage worker.
 * Included by swmm_drainage_driver.c after the untouched SWMM translation units.
 */
#ifndef SWMM_DRAINAGE_SNAPSHOT_H
#define SWMM_DRAINAGE_SNAPSHOT_H

typedef struct {
    unsigned char *data;
    size_t len;
    size_t cap;
    size_t pos;
    int err;
} Buf;

static void buf_init(Buf *b)
{
    b->data = NULL;
    b->len = b->cap = b->pos = 0;
    b->err = 0;
}

static void buf_free(Buf *b)
{
    free(b->data);
    b->data = NULL;
    b->len = b->cap = b->pos = 0;
}

static void buf_put(Buf *b, const void *src, size_t n)
{
    if (b->err || n == 0) return;
    if (b->len + n > b->cap) {
        size_t next = b->cap ? b->cap * 2 : 4096;
        while (next < b->len + n) next *= 2;
        unsigned char *p = (unsigned char *)realloc(b->data, next);
        if (!p) { b->err = 1; return; }
        b->data = p;
        b->cap = next;
    }
    memcpy(b->data + b->len, src, n);
    b->len += n;
}

static void buf_get(Buf *b, void *dst, size_t n)
{
    if (b->err || n == 0) return;
    if (b->pos + n > b->len) { b->err = 1; return; }
    memcpy(dst, b->data + b->pos, n);
    b->pos += n;
}

#define PUT(b, x) buf_put((b), &(x), sizeof(x))
#define GET(b, x) buf_get((b), &(x), sizeof(x))

static int table_cursor_index(const TTable *t)
{
    if (!t->thisEntry) return -1;
    int idx = 0;
    for (TTableEntry *e = t->firstEntry; e != NULL; e = e->next, ++idx) {
        if (e == t->thisEntry) return idx;
    }
    return -1;
}

static int table_cursor_restore(TTable *t, int idx)
{
    if (idx < 0) { t->thisEntry = NULL; return 1; }
    int cur = 0;
    for (TTableEntry *e = t->firstEntry; e != NULL; e = e->next, ++cur) {
        if (cur == idx) { t->thisEntry = e; return 1; }
    }
    return 0;
}

static void save_action_list(Buf *b, struct TAction *a)
{
    for (; a != NULL; a = a->next) {
        PUT(b, a->value);
        PUT(b, a->e1);
        PUT(b, a->e2);
    }
}

static void load_action_list(Buf *b, struct TAction *a)
{
    for (; a != NULL; a = a->next) {
        GET(b, a->value);
        GET(b, a->e1);
        GET(b, a->e2);
    }
}

static int capture_snapshot(Buf *b)
{
    uint32_t magic = 0x53574D31u;
    PUT(b, magic);
    PUT(b, Nobjects[NODE]);
    PUT(b, Nobjects[LINK]);
    PUT(b, Nnodes[STORAGE]);
    PUT(b, Nnodes[OUTFALL]);
    PUT(b, Nlinks[CONDUIT]);
    PUT(b, Nlinks[ORIFICE]);
    PUT(b, Nlinks[WEIR]);
    PUT(b, RuleCount);
    PUT(b, Nobjects[CURVE]);
    PUT(b, Nobjects[TSERIES]);

    PUT(b, OldRoutingTime); PUT(b, NewRoutingTime); PUT(b, RoutingDuration);
    PUT(b, ElapsedTime); PUT(b, ReportTime);
    PUT(b, TotalStepCount); PUT(b, ReportStepCount); PUT(b, NonConvergeCount);
    PUT(b, FlowError);

    PUT(b, NextEvent); PUT(b, BetweenEvents); PUT(b, NewRuleTime);

    PUT(b, VariableStep); PUT(b, Omega); PUT(b, Steps);
    if (Xnode && Nobjects[NODE] > 0) {
        buf_put(b, Xnode, (size_t)Nobjects[NODE] * sizeof(TXnode));
    }

    PUT(b, CurrentDate); PUT(b, CurrentTime); PUT(b, ControlValue); PUT(b, SetPoint);
    for (int r = 0; r < RuleCount; ++r) {
        save_action_list(b, Rules[r].thenActions);
        save_action_list(b, Rules[r].elseActions);
    }

    PUT(b, FlowTotals); PUT(b, StepFlowTotals); PUT(b, OldStepFlowTotals);
    if (NodeInflow && Nobjects[NODE] > 0) {
        buf_put(b, NodeInflow, (size_t)Nobjects[NODE] * sizeof(double));
        buf_put(b, NodeOutflow, (size_t)Nobjects[NODE] * sizeof(double));
    }

    for (int i = 0; i < Nobjects[NODE]; ++i) {
        TNode *n = &Node[i];
        PUT(b, n->degree); PUT(b, n->updated); PUT(b, n->crownElev);
        PUT(b, n->inflow); PUT(b, n->outflow); PUT(b, n->losses);
        PUT(b, n->oldVolume); PUT(b, n->newVolume); PUT(b, n->fullVolume);
        PUT(b, n->overflow); PUT(b, n->oldDepth); PUT(b, n->newDepth);
        PUT(b, n->oldLatFlow); PUT(b, n->newLatFlow);
        PUT(b, n->oldFlowInflow); PUT(b, n->oldNetInflow);
        PUT(b, n->qualInflow); PUT(b, n->apiExtInflow);
    }
    for (int i = 0; i < Nnodes[STORAGE]; ++i) {
        PUT(b, Storage[i].hrt); PUT(b, Storage[i].evapLoss); PUT(b, Storage[i].exfilLoss);
    }
    for (int i = 0; i < Nnodes[OUTFALL]; ++i) {
        PUT(b, Outfall[i].vRouted);
    }
    for (int i = 0; i < Nobjects[LINK]; ++i) {
        TLink *l = &Link[i];
        PUT(b, l->xsect);
        PUT(b, l->oldFlow); PUT(b, l->newFlow);
        PUT(b, l->oldDepth); PUT(b, l->newDepth);
        PUT(b, l->oldVolume); PUT(b, l->newVolume);
        PUT(b, l->surfArea1); PUT(b, l->surfArea2);
        PUT(b, l->qFull); PUT(b, l->setting); PUT(b, l->targetSetting);
        PUT(b, l->timeLastSet); PUT(b, l->froude); PUT(b, l->flowClass);
        PUT(b, l->dqdh); PUT(b, l->direction); PUT(b, l->bypassed);
        PUT(b, l->normalFlow); PUT(b, l->inletControl);
    }
    for (int i = 0; i < Nlinks[CONDUIT]; ++i) {
        TConduit *c = &Conduit[i];
        PUT(b, c->a1); PUT(b, c->a2);
        PUT(b, c->q1); PUT(b, c->q2);
        PUT(b, c->q1Old); PUT(b, c->q2Old);
        PUT(b, c->evapLossRate); PUT(b, c->seepLossRate);
        PUT(b, c->capacityLimited); PUT(b, c->superCritical);
        PUT(b, c->fullState);
    }
    for (int i = 0; i < Nlinks[ORIFICE]; ++i) {
        TOrifice *o = &Orifice[i];
        PUT(b, o->cOrif); PUT(b, o->hCrit); PUT(b, o->cWeir);
        PUT(b, o->length); PUT(b, o->surfArea);
    }
    for (int i = 0; i < Nlinks[WEIR]; ++i) {
        TWeir *w = &Weir[i];
        PUT(b, w->cSurcharge); PUT(b, w->length);
        PUT(b, w->slope); PUT(b, w->surfArea);
    }
    for (int i = 0; i < Nobjects[CURVE]; ++i) {
        int idx = table_cursor_index(&Curve[i]);
        PUT(b, Curve[i].x1); PUT(b, Curve[i].x2);
        PUT(b, Curve[i].y1); PUT(b, Curve[i].y2);
        PUT(b, idx);
    }
    for (int i = 0; i < Nobjects[TSERIES]; ++i) {
        int idx = table_cursor_index(&Tseries[i]);
        PUT(b, Tseries[i].x1); PUT(b, Tseries[i].x2);
        PUT(b, Tseries[i].y1); PUT(b, Tseries[i].y2);
        PUT(b, idx);
    }
    return !b->err;
}

static int restore_snapshot(Buf *b)
{
    uint32_t magic = 0;
    int counts[10];
    GET(b, magic);
    for (int i = 0; i < 10; ++i) GET(b, counts[i]);
    if (b->err || magic != 0x53574D31u) return 0;
    if (counts[0] != Nobjects[NODE] || counts[1] != Nobjects[LINK]
        || counts[2] != Nnodes[STORAGE] || counts[3] != Nnodes[OUTFALL]
        || counts[4] != Nlinks[CONDUIT] || counts[5] != Nlinks[ORIFICE]
        || counts[6] != Nlinks[WEIR] || counts[7] != RuleCount
        || counts[8] != Nobjects[CURVE] || counts[9] != Nobjects[TSERIES]) {
        return 0;
    }

    GET(b, OldRoutingTime); GET(b, NewRoutingTime); GET(b, RoutingDuration);
    GET(b, ElapsedTime); GET(b, ReportTime);
    GET(b, TotalStepCount); GET(b, ReportStepCount); GET(b, NonConvergeCount);
    GET(b, FlowError);

    GET(b, NextEvent); GET(b, BetweenEvents); GET(b, NewRuleTime);

    GET(b, VariableStep); GET(b, Omega); GET(b, Steps);
    if (Xnode && Nobjects[NODE] > 0) {
        buf_get(b, Xnode, (size_t)Nobjects[NODE] * sizeof(TXnode));
    }

    GET(b, CurrentDate); GET(b, CurrentTime); GET(b, ControlValue); GET(b, SetPoint);
    for (int r = 0; r < RuleCount; ++r) {
        load_action_list(b, Rules[r].thenActions);
        load_action_list(b, Rules[r].elseActions);
    }

    GET(b, FlowTotals); GET(b, StepFlowTotals); GET(b, OldStepFlowTotals);
    if (NodeInflow && Nobjects[NODE] > 0) {
        buf_get(b, NodeInflow, (size_t)Nobjects[NODE] * sizeof(double));
        buf_get(b, NodeOutflow, (size_t)Nobjects[NODE] * sizeof(double));
    }

    for (int i = 0; i < Nobjects[NODE]; ++i) {
        TNode *n = &Node[i];
        GET(b, n->degree); GET(b, n->updated); GET(b, n->crownElev);
        GET(b, n->inflow); GET(b, n->outflow); GET(b, n->losses);
        GET(b, n->oldVolume); GET(b, n->newVolume); GET(b, n->fullVolume);
        GET(b, n->overflow); GET(b, n->oldDepth); GET(b, n->newDepth);
        GET(b, n->oldLatFlow); GET(b, n->newLatFlow);
        GET(b, n->oldFlowInflow); GET(b, n->oldNetInflow);
        GET(b, n->qualInflow); GET(b, n->apiExtInflow);
    }
    for (int i = 0; i < Nnodes[STORAGE]; ++i) {
        GET(b, Storage[i].hrt); GET(b, Storage[i].evapLoss); GET(b, Storage[i].exfilLoss);
    }
    for (int i = 0; i < Nnodes[OUTFALL]; ++i) {
        GET(b, Outfall[i].vRouted);
    }
    for (int i = 0; i < Nobjects[LINK]; ++i) {
        TLink *l = &Link[i];
        GET(b, l->xsect);
        GET(b, l->oldFlow); GET(b, l->newFlow);
        GET(b, l->oldDepth); GET(b, l->newDepth);
        GET(b, l->oldVolume); GET(b, l->newVolume);
        GET(b, l->surfArea1); GET(b, l->surfArea2);
        GET(b, l->qFull); GET(b, l->setting); GET(b, l->targetSetting);
        GET(b, l->timeLastSet); GET(b, l->froude); GET(b, l->flowClass);
        GET(b, l->dqdh); GET(b, l->direction); GET(b, l->bypassed);
        GET(b, l->normalFlow); GET(b, l->inletControl);
    }
    for (int i = 0; i < Nlinks[CONDUIT]; ++i) {
        TConduit *c = &Conduit[i];
        GET(b, c->a1); GET(b, c->a2);
        GET(b, c->q1); GET(b, c->q2);
        GET(b, c->q1Old); GET(b, c->q2Old);
        GET(b, c->evapLossRate); GET(b, c->seepLossRate);
        GET(b, c->capacityLimited); GET(b, c->superCritical);
        GET(b, c->fullState);
    }
    for (int i = 0; i < Nlinks[ORIFICE]; ++i) {
        TOrifice *o = &Orifice[i];
        GET(b, o->cOrif); GET(b, o->hCrit); GET(b, o->cWeir);
        GET(b, o->length); GET(b, o->surfArea);
    }
    for (int i = 0; i < Nlinks[WEIR]; ++i) {
        TWeir *w = &Weir[i];
        GET(b, w->cSurcharge); GET(b, w->length);
        GET(b, w->slope); GET(b, w->surfArea);
    }
    for (int i = 0; i < Nobjects[CURVE]; ++i) {
        int idx = -1;
        GET(b, Curve[i].x1); GET(b, Curve[i].x2);
        GET(b, Curve[i].y1); GET(b, Curve[i].y2);
        GET(b, idx);
        if (!table_cursor_restore(&Curve[i], idx)) return 0;
    }
    for (int i = 0; i < Nobjects[TSERIES]; ++i) {
        int idx = -1;
        GET(b, Tseries[i].x1); GET(b, Tseries[i].x2);
        GET(b, Tseries[i].y1); GET(b, Tseries[i].y2);
        GET(b, idx);
        if (!table_cursor_restore(&Tseries[i], idx)) return 0;
    }
    return !b->err && b->pos == b->len;
}

#endif /* SWMM_DRAINAGE_SNAPSHOT_H */
