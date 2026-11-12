//
// PyHexen - Python reimplementation of the Hexen engine.
// Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
// released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
//
// Per-tic trace hook for Chocolate Hexen (T15.6). build_trace.sh appends this file
// to src/hexen/p_tick.c and calls PYH_TraceTic() right after `leveltime++` in
// P_Ticker. The output format must match pyhexen.sim.demo_trace exactly.
//
// Env vars:
//   PYHEXEN_TRACE       path of the trace file (no trace when unset)
//   PYHEXEN_TRACE_DUMP  comma-separated leveltimes at which every mobj is dumped
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int prndindex;

static FILE *pyh_trace_file;
static int pyh_trace_init;
static char pyh_dump_tics[512];

static int PYH_DumpThisTic(int tic)
{
    char buf[16];
    const char *p = pyh_dump_tics;
    int n;

    while (*p)
    {
        n = 0;
        while (*p && *p != ',' && n < 15)
        {
            buf[n++] = *p++;
        }
        buf[n] = 0;
        if (*p == ',')
        {
            p++;
        }
        if (n > 0 && atoi(buf) == tic)
        {
            return 1;
        }
    }
    return 0;
}

static void PYH_TraceTic(void)
{
    thinker_t *th;
    mobj_t *mo;
    mobj_t *pmo;
    unsigned int h = 0;
    unsigned int sh = 0;
    int count = 0;
    int dump;
    int i;

    if (!pyh_trace_init)
    {
        const char *path = getenv("PYHEXEN_TRACE");
        const char *dumps = getenv("PYHEXEN_TRACE_DUMP");

        pyh_trace_init = 1;
        if (path)
        {
            pyh_trace_file = fopen(path, "w");
        }
        if (dumps)
        {
            strncpy(pyh_dump_tics, dumps, sizeof(pyh_dump_tics) - 1);
        }
    }
    if (!pyh_trace_file)
    {
        return;
    }

    dump = PYH_DumpThisTic(leveltime);
    for (th = thinkercap.next; th != &thinkercap; th = th->next)
    {
        // Blaster missiles (Mage wand, Cleric flame) run P_BlasterMobjThinker.
        if (th->function != P_MobjThinker
         && th->function != P_BlasterMobjThinker)
        {
            continue;
        }
        mo = (mobj_t *) th;
        h = (h * 1000003u)
            ^ ((unsigned int) mo->x + 3u * (unsigned int) mo->y
               + 7u * (unsigned int) mo->z + 11u * (unsigned int) mo->type
               + 13u * (unsigned int) mo->health
               + 17u * (unsigned int) (mo->state - states));
        count++;
        if (dump)
        {
            fprintf(pyh_trace_file, "M %d %d %d %d %d %d %d %u %d %d %d %d\n",
                    mo->type, mo->x, mo->y, mo->z, mo->momx, mo->momy,
                    mo->momz, (unsigned int) mo->angle, mo->health,
                    (int) (mo->state - states), mo->tics, mo->flags);
        }
    }

    // Sector hash over floor, ceiling and light level, in sector order.
    for (i = 0; i < numsectors; i++)
    {
        sh = (sh * 1000003u)
            ^ ((unsigned int) sectors[i].floorheight
               + 3u * (unsigned int) sectors[i].ceilingheight
               + 7u * (unsigned int) sectors[i].lightlevel);
        if (dump)
        {
            fprintf(pyh_trace_file, "S %d %d %d %d\n", i, sectors[i].floorheight,
                    sectors[i].ceilingheight, sectors[i].lightlevel);
        }
    }

    pmo = players[0].mo;
    fprintf(pyh_trace_file, "T %d %d %d %d %d %u %d %d %d %d %d %08x %08x\n",
            leveltime, prndindex,
            pmo ? pmo->x : 0, pmo ? pmo->y : 0, pmo ? pmo->z : 0,
            pmo ? (unsigned int) pmo->angle : 0u,
            pmo ? pmo->momx : 0, pmo ? pmo->momy : 0, pmo ? pmo->momz : 0,
            players[0].health, count, h, sh);
    fflush(pyh_trace_file);
}
