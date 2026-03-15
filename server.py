# server.py — SAGAscape visualisatie met werkende grafieken
# Strategie: gebruik CanvasGrid voor kaart + schrijf eigen JS-grafiek
# die direct in de Mesa templates-folder geïnjecteerd wordt

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider, Checkbox
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from simulation import SAGAscapeModel
from resources import PatchAgent
from communities import CommunityAgent


# =========================================================================
# AGENT PORTRAYAL  (kleuren identiek aan NetLogo)
# =========================================================================

def agent_portrayal(agent):
    if isinstance(agent, CommunityAgent):
        active = getattr(agent, 'active', True)
        stype  = getattr(agent, 'settlement_type', 'village')
        pop    = getattr(agent, 'population', 100)
        # Huisje-grootte: zichtbaar per settlement type
        sizes  = {"hamlet": 1, "village": 2, "town": 3}
        size   = sizes.get(stype, 2)
        colors = {"hamlet": "#8B008B", "village": "#FF6600", "town": "#FF0000"}
        color  = "#888888" if not active else colors.get(stype, "#FF6600")
        return {"Shape": "rect", "Color": color, "Filled": "true",
                "Layer": 1, "w": size, "h": size}

    elif isinstance(agent, PatchAgent):
        cell = agent.model.grid.get_cell_list_contents([agent.pos])
        if any(isinstance(a, CommunityAgent) for a in cell):
            return {}

        # ── WATER ─────────────────────────────────────────────────────────────
        if not agent.land:
            return {"Shape": "rect", "Color": "#4169B8",
                    "Filled": "true", "Layer": 0, "w": 1, "h": 1}

        # ── BRAND (net uitgebroken) ────────────────────────────────────────────
        if getattr(agent, 'time_since_fire', 999) == 0:
            return {"Shape": "rect", "Color": "#FF4500",
                    "Filled": "true", "Layer": 0, "w": 1, "h": 1}

        # ── KLEIGROEVE ────────────────────────────────────────────────────────
        if agent.land_is == "clay quarry":
            return {"Shape": "rect", "Color": "#888888",
                    "Filled": "true", "Layer": 0, "w": 1, "h": 1}

        # Globale max houtstock eenmalig cachen op model voor consistente schaal
        if not hasattr(agent.model, '_global_wood_max') or agent.model._global_wood_max is None:
            all_max = [a.wood_maxStandingStock
                       for a in agent.model.schedule.agents
                       if isinstance(a, PatchAgent) and a.wood_maxStandingStock > 0]
            agent.model._global_wood_max = max(all_max) if all_max else 1.0
        global_max = agent.model._global_wood_max

        # ── AKKER (wood?=False, food?=True) ───────────────────────────────────
        # NetLogo landuse-viz: palette:scale-gradient [[153 52 4][254 217 142]]
        #   food-fertility f-max f-min
        # t=1 → donkerbruin/oranje (vruchtbaar)   t=0 → lichtgeel/beige (net geoogst)
        # Minimum t=0.15 zodat geoogste patches zichtbaar BEIGE blijven (niet wit)
        # en duidelijk onderscheidbaar zijn van de groene achtergrond
        if not agent.wood_regeneration and agent.food_regeneration:
            max_f = agent.original_food_value if agent.original_food_value > 0 else 1.0
            # Minimumkleur: t=0.15 = lichtbeige (geoogst maar zichtbaar anders dan bos)
            t = max(0.15, min(1.0, agent.food_fertility / max_f))
            r = int(254 - t * 101)   # 254 (beige) → 153 (bruin)
            g = int(217 - t * 165)   # 217         →  52
            b = int(142 - t * 138)   # 142         →   4
            color = f"rgb({r},{g},{b})"
            return {"Shape": "rect", "Color": color,
                    "Filled": "true", "Layer": 0, "w": 1, "h": 1}

        # ── BOS (wood?=True, stock > 0) ───────────────────────────────────────
        # NetLogo landuse-viz: palette:scale-gradient [[0 109 44][186 228 179]]
        #   wood-standingStock w-max w-min
        # t=1 → donkergroen (vol/oud bos)   t=0 → lichtgroen (jong/leeg bos)
        if agent.wood_regeneration and agent.wood_standingStock > 0:
            t = min(1.0, max(0.0, agent.wood_standingStock / global_max))
            r = int(186 - t * 186)   # 186 → 0
            g = int(228 - t * 119)   # 228 → 109
            b = int(179 - t * 135)   # 179 → 44
            color = f"rgb({r},{g},{b})"
            return {"Shape": "rect", "Color": color,
                    "Filled": "true", "Layer": 0, "w": 1, "h": 1}

        # ── OPEN BOS (wood?=True, stock=0): herstelperiode na brand/kap ───────
        # Laagste boskleur = meest lichtgroen
        if agent.wood_regeneration and agent.wood_standingStock == 0:
            return {"Shape": "rect", "Color": "rgb(186,228,179)",
                    "Filled": "true", "Layer": 0, "w": 1, "h": 1}

        # ── TOPOGRAFISCHE FALLBACK (ongebruikt land, geen food/wood state) ────
        # NetLogo basis-visualisatie: palette:scale-gradient
        #   [[0 104 55][255 255 191][252 141 89]] elevation e-min e-max
        # Laag (0m) → donkergroen   Midden (~1500m) → lichtgeel   Hoog (2500m+) → oranje
        elev = getattr(agent, 'elevation', 0) or 0
        if elev != elev or elev < 0:
            elev = 0
        t = min(1.0, max(0.0, elev / 2500.0))
        if t < 0.5:
            s = t * 2  # 0→1 voor lage helft
            r = int(0   + s * 255)   #   0 → 255
            g = int(104 + s * 151)   # 104 → 255
            b = int(55  + s * 136)   #  55 → 191
        else:
            s = (t - 0.5) * 2  # 0→1 voor hoge helft
            r = int(255 - s * 3)     # 255 → 252
            g = int(255 - s * 114)   # 255 → 141
            b = int(191 - s * 102)   # 191 →  89
        color = f"rgb({r},{g},{b})"
        return {"Shape": "rect", "Color": color,
                "Filled": "true", "Layer": 0, "w": 1, "h": 1}

    return {}




# =========================================================================
# GRIDGROOTTE
# =========================================================================

try:
    from gis import load_raster
    _elev, _ = load_raster(os.path.join(_HERE, "data", "Altitude.asc"))
    _fh, _fw = _elev.shape
    _SCALE   = 2   # ← zet gelijk aan SCALE in environment.py
    _GW      = _fw // _SCALE
    _GH      = _fh // _SCALE
    print(f"Server: origineel raster {_fw}x{_fh}, grid (x{_SCALE}): {_GW}x{_GH}")
except Exception as e:
    _GW, _GH = 50, 50
    print(f"Server: GIS niet gevonden ({e}), standaard 50x50.")

_CW = 800
_CH = 400

# =========================================================================
# ELEMENTEN & SERVER
# =========================================================================

grid      = CanvasGrid(agent_portrayal, _GW, _GH, _CW, _CH)

model_params = {
    "width":  _GW, "height": _GH,
    "food_demand_pc":        Slider("Food demand/capita",    1.15, 0.75, 1.45, 0.05),
    "wood_demand_pc":        Slider("Wood demand/capita",    2.35, 1,    5,    0.05),
    "clay_demand_pc":        Slider("Clay demand/capita",    6,    0,    10,   1),
    "active_percentage":     Slider("Active %",              10,   0,    100,  1),
    "agriculture_days":      Slider("Agriculture days",      250,  120,  265,  5),
    "grain_per_grain_yield": Slider("Grain/grain yield",     6,    2,    6,    0.5),
    "regeneration_time":     Slider("Regeneration time",     2,    1,    3,    1),
    "kgs_wood_per_kg_clay":  Slider("Kg wood/kg clay",       0.29, 0.14, 0.39, 0.05),
    "clay_threshold":        Slider("Clay threshold",        0.5,  0.3,  0.5,  0.05),
    "territory":             Slider("Territory (px radius)", 25,   0,    200,  1),
    "bad_harvest_interval":  Slider("Bad harvest interval",  5,    1,    10,   1),
    "forest_regrowth_lag":   Slider("Forest regrowth lag",   6,    3,    10,   1),
    "time_limit":            Slider("Time limit",            1000, 0,    3000, 1),
    "landuse_visualization": Checkbox("Landuse visualization", True),
}

server = ModularServer(
    SAGAscapeModel,
    [grid,],
    "SAGAscape Community Simulation",
    model_params
)
server.port = 8521

def launch():
    server.launch()