# patches & landscape logic

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from communities import CommunityAgent
from resources import PatchAgent, adapted_fertility
import random
import heapq
import numpy as np
from mesa.datacollection import DataCollector
from gis import load_raster, load_shapefile
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def _data(filename):
    return os.path.join(DATA_DIR, filename)


class SAGAscapeModel(Model):
    def __init__(self, width=50, height=50, num_communities=5,
                 food_demand_pc=1.15, active_percentage=10,
                 agriculture_days=250, wood_demand_pc=2.35, regeneration_time=2,
                 clay_demand_pc=6, grain_per_grain_yield=6,
                 kgs_wood_per_kg_clay=0.29, clay_threshold=0.5,
                 territory=50, bad_harvest_interval=5,
                 forest_regrowth_lag=6, time_limit=1000, landuse_visualization=True):
        super().__init__()

        # Globale variabelen
        self.regeneration_reserve  = 0.1
        self.burn_size             = []
        self.bad_harvest_modifier  = 1

        # Parameters
        self.food_demand_pc        = float(food_demand_pc)
        self.active_percentage     = float(active_percentage)
        self.agriculture_days      = int(agriculture_days)
        self.wood_demand_pc        = float(wood_demand_pc)
        self.regeneration_time     = float(regeneration_time)
        self.clay_demand_pc        = float(clay_demand_pc)
        self.grain_per_grain_yield = float(grain_per_grain_yield)
        self.kgs_wood_per_kg_clay  = float(kgs_wood_per_kg_clay)
        self.clay_threshold        = float(clay_threshold)
        self.territory             = float(territory)
        self.bad_harvest_interval  = float(bad_harvest_interval)
        self.forest_regrowth_lag   = int(forest_regrowth_lag)
        self.time_limit            = int(time_limit)
        self.landuse_visualization = landuse_visualization

        # === GIS data laden ===
        self.raster_transform = None
        try:
            self.elev_data,       self.raster_transform = load_raster(_data("Altitude.asc"))
            self.fert_data,       _ = load_raster(_data("Fertility_K.asc"))
            self.forest_max_data, _ = load_raster(_data("ForestMax.asc"))
            self.forest_a_data,   _ = load_raster(_data("ForestA.asc"))
            self.forest_b_data,   _ = load_raster(_data("ForestB.asc"))
            self.clay_data,       _ = load_raster(_data("Clay content_kg_per_ha.asc"))
            self.walking_data,    _ = load_raster(_data("Tobler_EPSG32636.asc"))
            self.water_data,      _ = load_raster(_data("lakesAndRiversRasterized.asc"))

            full_height, full_width = self.elev_data.shape
            SCALE = 2
            self.scale    = SCALE
            raster_width  = full_width  // SCALE
            raster_height = full_height // SCALE

            self.elev_data       = self.elev_data      [::SCALE, ::SCALE]
            self.fert_data       = self.fert_data      [::SCALE, ::SCALE]
            self.forest_max_data = self.forest_max_data[::SCALE, ::SCALE]
            self.forest_a_data   = self.forest_a_data  [::SCALE, ::SCALE]
            self.forest_b_data   = self.forest_b_data  [::SCALE, ::SCALE]
            self.clay_data       = self.clay_data      [::SCALE, ::SCALE]
            self.walking_data    = self.walking_data   [::SCALE, ::SCALE]
            self.water_data      = self.water_data     [::SCALE, ::SCALE]

            width  = raster_width
            height = raster_height
            print(f"GIS rasterdata succesvol geladen.")
            print(f"Origineel raster: {full_width}×{full_height}, na downsampling (×{SCALE}): {width}×{height}")

        except Exception as e:
            print(f"Fout bij laden GIS data: {e}")
            self.scale           = 2
            self.elev_data       = np.random.uniform(0, 1000, (height, width))
            self.fert_data       = np.random.uniform(0, 3.5,  (height, width))
            self.forest_max_data = np.random.uniform(50, 200, (height, width))
            self.forest_a_data   = np.random.uniform(1, 50,   (height, width))
            self.forest_b_data   = np.random.uniform(0.01, 0.1, (height, width))
            self.clay_data       = np.random.uniform(0, 5000, (height, width))
            self.walking_data    = np.ones((height, width))
            self.water_data      = np.zeros((height, width))

        self.grid     = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)

        self.setup_topo(width, height)
        self.setup_communities(num_communities)
        self.setup_least_cost_distances()
        self.initial_periodization()
        self.setup_resources()
        self.setup_regeneration()

        # Gecachede lijsten
        self.patch_map = {
            a.pos: a for a in self.schedule.agents if isinstance(a, PatchAgent)
        }
        self.land_patches = [
            a for a in self.schedule.agents if isinstance(a, PatchAgent) and a.land
        ]
        self.active_communities = [
            a for a in self.schedule.agents
            if isinstance(a, CommunityAgent) and a.active
        ]

        # Render-cache: numpy arrays voor snelle visualisatie
        # Worden gevuld door _init_render_cache() in server.py bij eerste render
        # en bijgewerkt via _update_render_cache() na elke stap
        self.rc_ready = False

        self.datacollector = DataCollector(
            model_reporters={
                "total_food_stock": lambda m: sum(
                    max(0, c.food_stock) for c in m.schedule.agents
                    if isinstance(c, CommunityAgent) and c.active),
                "total_wood_stock": lambda m: sum(
                    max(0, c.wood_stock) for c in m.schedule.agents
                    if isinstance(c, CommunityAgent) and c.active),
                "total_clay_stock": lambda m: sum(
                    max(0, c.clay_stock) for c in m.schedule.agents
                    if isinstance(c, CommunityAgent) and c.active),
                "total_population": lambda m: sum(
                    c.population for c in m.schedule.agents
                    if isinstance(c, CommunityAgent) and c.active),
                "active_communities": lambda m: sum(
                    1 for c in m.schedule.agents
                    if isinstance(c, CommunityAgent) and c.active),
            }
        )
        self.running = True

    # =========================================================================
    # SETUP TOPO
    # =========================================================================

    def setup_topo(self, width, height):
        def safe(val, default=0.0, positive=False):
            v = float(val)
            if v != v or v < 0:
                return default
            if positive and v == 0:
                return default
            return v

        for x in range(width):
            for y in range(height):
                patch = PatchAgent(f"patch_{x}_{y}", self)
                py = (height - 1) - y

                patch.elevation             = safe(self.elev_data[py, x],        0.0)
                patch.wood_maxStandingStock = safe(self.forest_max_data[py, x],  0.0)
                patch.walkingTime           = safe(self.walking_data[py, x],     1.0, positive=True)
                patch.wood_rico             = safe(self.forest_a_data[py, x],    1.0, positive=True)
                patch.wood_power            = safe(self.forest_b_data[py, x],    0.01, positive=True)
                patch.clay_quantity         = safe(self.clay_data[py, x],        0.0) / 1000

                water_val = float(self.water_data[py, x])
                if water_val == 0:
                    patch.land = True
                    patch.land_is = "open land"
                    if patch.elevation < 1100:
                        patch.fire_return_rate = 3 + self.random.randint(0, 25)
                    else:
                        patch.fire_return_rate = round((patch.elevation - 1100) * 171 / 300) + 3 + self.random.randint(0, 25)
                    patch.time_since_fire = self.random.randint(0, patch.fire_return_rate)
                else:
                    patch.land = False
                    patch.land_is = "water"
                    patch.wood_regeneration = False
                    patch.food_regeneration = False
                    patch.clay_source       = False
                    patch.clay_quantity     = 0
                    patch.wood_maxStandingStock = 0

                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

        self.burn_size = []

    # =========================================================================
    # SETUP COMMUNITIES
    # =========================================================================

    def setup_communities(self, num_communities_fallback):
        communities_created = 0
        try:
            sites = load_shapefile(_data("sagascape-sites-EPSG32636.shp"))

            if self.raster_transform is not None:
                t = self.raster_transform
                px_w     = t.a
                px_h     = abs(t.e)
                x_origin = t.c + 0.5 * px_w
                y_origin = t.f - 0.5 * px_h
            else:
                x_origin = 0
                y_origin = self.grid.height * self.scale
                px_w = 1
                px_h = 1

            for site in sites:
                utm_x           = site.get("x")
                utm_y           = site.get("y")
                site_name       = site.get("Site",  f"site_{communities_created}")
                settlement_type = site.get("Type",  "village")
                start_period    = site.get("Start", "IA")

                if settlement_type == "hamlet":
                    population = max(1, round(random.gauss(50, 10)))
                elif settlement_type == "village":
                    population = max(1, round(random.gauss(500, 100)))
                elif settlement_type == "town":
                    population = max(1, round(random.gauss(1000, 200)))
                else:
                    population = 100

                gx = int((utm_x - x_origin) / px_w / self.scale)
                gy = int((y_origin - utm_y) / px_h / self.scale)
                gx = min(max(gx, 0), self.grid.width  - 1)
                gy = min(max(gy, 0), self.grid.height - 1)

                patch_here = self._get_patch(gx, gy)
                elev = patch_here.elevation if patch_here else 0

                agent = CommunityAgent(
                    communities_created, self,
                    site_name=site_name,
                    population=population,
                    elevation_here=elev,
                    settlement_type=settlement_type,
                    start_period=start_period
                )
                agent.active = True
                self.schedule.add(agent)
                self.grid.place_agent(agent, (gx, gy))
                communities_created += 1

            print(f"{communities_created} communities geladen uit shapefile.")

        except Exception as e:
            print(f"Shapefile niet beschikbaar ({e}) — willekeurige communities aanmaken.")
            for i in range(num_communities_fallback):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                patch_here = self._get_patch(x, y)
                elev = patch_here.elevation if patch_here else 0
                agent = CommunityAgent(i, self, site_name=f"community_{i}",
                                       population=100, elevation_here=elev,
                                       settlement_type="village", start_period="IA")
                agent.active = True
                self.schedule.add(agent)
                self.grid.place_agent(agent, (x, y))

    # =========================================================================
    # SETUP RESOURCES
    # =========================================================================

    def setup_resources(self):
        for agent in self.schedule.agents:
            if not isinstance(agent, PatchAgent) or not agent.land:
                continue

            patch_has_community = any(
                isinstance(a, CommunityAgent)
                for a in self.grid.get_cell_list_contents([agent.pos])
            )

            if not patch_has_community:
                agent.wood_regeneration = True
                agent.food_regeneration = True
                agent.wood_age = 200 + self.random.randint(0, 200)

                raw_fert = float(self.fert_data[(self.grid.height - 1) - agent.pos[1], agent.pos[0]])
                agent.food_fertility = adapted_fertility(raw_fert)
                agent.time_since_abandonment = 0
                agent.clay_source = agent.clay_quantity > self.clay_threshold * 10000 * 2
            else:
                agent.wood_maxStandingStock = 0
                agent.food_fertility        = 0
                agent.wood_regeneration     = False
                agent.food_regeneration     = False
                agent.clay_source           = False

            agent.update_wood_standing_stock()
            agent.wood_standingStock = min(agent.wood_standingStock, agent.wood_maxStandingStock)
            agent.original_food_value = agent.food_fertility

        # ── Spinup ────────────────────────────────────────────────────────────
        # NetLogo: ask communities [...]   →  alleen ACTIEVE communities (breed 'communities',
        #          niet 'inactive-communities')
        # NetLogo: patches with [wood-standingStock > 0]  →  het VOLLEDIGE grid, niet beperkt
        #          tot het territorium. Dit geeft de grotere en minder scherpe rand in NetLogo.
        # NetLogo: [distance homebase]  →  Euclidische pixelafstand
        # NetLogo: total-population = sum [population] of communities  →  enkel actieve

        active_communities = [a for a in self.schedule.agents
                              if isinstance(a, CommunityAgent) and a.active]
        total_land         = sum(1 for a in self.schedule.agents
                                 if isinstance(a, PatchAgent) and a.land)
        total_population   = sum(c.population for c in active_communities) or 1

        # Verzamel alle bospatches op het volledige grid (zoals NetLogo)
        all_wooded = [a for a in self.schedule.agents
                      if isinstance(a, PatchAgent) and a.land and a.wood_standingStock > 0]

        if all_wooded:
            wooded_pos = np.array([p.pos for p in all_wooded])

            for community in active_communities:
                # Aantal te ontbossen patches, gewogen naar bevolkingsaandeel
                initial_open = int(0.60 * total_land * community.population / total_population)
                if initial_open == 0:
                    continue

                cx, cy = community.pos
                # Euclidische pixelafstand — identiek aan NetLogo's `distance homebase`
                dists       = (wooded_pos[:, 0] - cx) ** 2 + (wooded_pos[:, 1] - cy) ** 2
                nearest_idx = np.argsort(dists)[:initial_open]
                for idx in nearest_idx:
                    p = all_wooded[idx]
                    p.wood_standingStock = 0
                    p.wood_age           = 0
                    # NetLogo spinup: patch wordt ontbost maar is nog GEEN akker.
                    # wood? = true (bos mag terugkomen), stock = 0
                    # pas als een community hem echt exploiteert wordt wood?=false, food?=true
                    # Dit geeft de lichtgroene spinup-cirkel zoals in NetLogo
                    p.wood_regeneration  = True
                    p.food_regeneration  = False
                    p.land_is            = "open land"

        self.bad_harvest_modifier = 1  # NetLogo: set bad-harvest-modifier 1

    # =========================================================================
    # SETUP LEAST-COST DISTANCES
    # =========================================================================

    def setup_least_cost_distances(self):
        territory = float(self.territory)

        patch_map = {
            a.pos: a for a in self.schedule.agents if isinstance(a, PatchAgent)
        }

        for p in patch_map.values():
            p.in_range_of  = []
            p.claimed_cost = []

        for community in self.schedule.agents:
            if not isinstance(community, CommunityAgent):
                continue

            cx, cy = community.pos
            # NetLogo: "in-radius territory" = Euclidische straal in pixels
            # territory=50 → 50 pixels = 50km bij 1km/pixel
            # Bij scale=2: pixels zijn 2km → straal = territory/scale pixels
            radius_px = territory / self.scale
            radius_sq = radius_px ** 2

            dist   = {}
            heap   = [(0.0, (cx, cy))]

            while heap:
                cost, pos = heapq.heappop(heap)
                if pos in dist:
                    continue
                dist[pos] = cost

                px, py = pos
                current_patch = patch_map.get(pos)
                w_current = current_patch.walkingTime if current_patch and current_patch.walkingTime > 0 else 1.0

                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nx, ny = px + dx, py + dy
                    if not (0 <= nx < self.grid.width and 0 <= ny < self.grid.height):
                        continue
                    if (nx, ny) in dist:
                        continue
                    # Euclidische grens (zoals NetLogo in-radius)
                    if (nx-cx)**2 + (ny-cy)**2 > radius_sq:
                        continue
                    neighbor = patch_map.get((nx, ny))
                    if neighbor is None:
                        continue
                    w_neighbor  = neighbor.walkingTime if neighbor.walkingTime > 0 else 1.0
                    link_weight = 0.5 * (w_current + w_neighbor)
                    heapq.heappush(heap, (cost + link_weight, (nx, ny)))

            community.candidate_patches = []
            community.cost_cache       = {}
            for pos, lcd in dist.items():
                patch = patch_map.get(pos)
                if patch and patch.land:
                    has_comm = any(isinstance(a, CommunityAgent)
                                   for a in self.grid.get_cell_list_contents([pos]))
                    if not has_comm:
                        community.candidate_patches.append(patch)
                        community.cost_cache[id(patch)] = lcd

            # Sortering gebeurt elke tick in exploit_resources (zoals NetLogo)
            # zodat de actuele patch-waarden (fertility, stock) de volgorde bepalen.
            # _cost_cache bevat de least-cost afstanden voor snelle O(1) opzoekingen.

        # Debug: rapporteer communities zonder kandidaat-patches
        no_candidates = []
        for a in self.schedule.agents:
            if isinstance(a, CommunityAgent):
                if len(a.candidate_patches) == 0:
                    # Probeer community te verplaatsen naar dichtstbijzijnde land-patch
                    cx, cy = a.pos
                    best_patch = None
                    best_dist  = float('inf')
                    for p in patch_map.values():
                        if p.land:
                            d = (p.pos[0]-cx)**2 + (p.pos[1]-cy)**2
                            if d < best_dist:
                                best_dist = d
                                best_patch = p
                    if best_patch and best_dist > 0:
                        # Verplaats community naar land-patch en herbereken
                        self.grid.move_agent(a, best_patch.pos)
                        no_candidates.append(
                            f"{getattr(a,'site_name',a.unique_id)} verplaatst "
                            f"({cx},{cy})→{best_patch.pos}"
                        )
                    else:
                        no_candidates.append(
                            f"{getattr(a,'site_name',a.unique_id)} @ ({cx},{cy}): "
                            f"geen land-patch gevonden!"
                        )

        if no_candidates:
            print(f"  !! Communities zonder kandidaat-patches: {no_candidates}")
            # Herbereken LCD voor verplaatste communities
            for a in self.schedule.agents:
                if isinstance(a, CommunityAgent) and len(a.candidate_patches) == 0:
                    cx, cy = a.pos
                    radius_px = territory / self.scale
                    radius_sq = radius_px ** 2
                    dist2 = {}
                    heap2 = [(0.0, (cx, cy))]
                    while heap2:
                        cost, pos = __import__('heapq').heappop(heap2)
                        if pos in dist2: continue
                        dist2[pos] = cost
                        px2, py2 = pos
                        cp = patch_map.get(pos)
                        wc = cp.walkingTime if cp and cp.walkingTime > 0 else 1.0
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nx2, ny2 = px2+dx, py2+dy
                            if not (0<=nx2<self.grid.width and 0<=ny2<self.grid.height): continue
                            if (nx2,ny2) in dist2: continue
                            if (nx2-cx)**2+(ny2-cy)**2 > radius_sq: continue
                            nb = patch_map.get((nx2,ny2))
                            if nb is None: continue
                            wn = nb.walkingTime if nb.walkingTime > 0 else 1.0
                            __import__('heapq').heappush(heap2, (cost + 0.5*(wc+wn), (nx2,ny2)))
                    for pos2, lcd2 in dist2.items():
                        p2 = patch_map.get(pos2)
                        if p2 and p2.land:
                            hc = any(isinstance(ag, CommunityAgent)
                                     for ag in self.grid.get_cell_list_contents([pos2]))
                            if not hc:
                                a.candidate_patches.append(p2)
                                a.cost_cache[id(p2)] = lcd2

        total_comm  = sum(1 for a in self.schedule.agents if isinstance(a, CommunityAgent))
        with_cands  = sum(1 for a in self.schedule.agents
                          if isinstance(a, CommunityAgent) and len(a.candidate_patches) > 0)
        print(f"Least-cost distances: {total_comm} communities, "
              f"{with_cands} met kandidaat-patches.")

    # =========================================================================
    # INITIAL PERIODIZATION
    # =========================================================================

    def initial_periodization(self):
        active_count   = 0
        inactive_count = 0
        for agent in self.schedule.agents:
            if isinstance(agent, CommunityAgent):
                if agent.start_period == "IA":
                    agent.active = True
                    active_count += 1
                else:
                    agent.active = False
                    inactive_count += 1
        print(f"Periodisering: {active_count} actieve (IA) en {inactive_count} inactieve communities.")

    # =========================================================================
    # SETUP REGENERATION
    # =========================================================================

    def setup_regeneration(self):
        reserve = self.regeneration_reserve
        for agent in self.schedule.agents:
            if not isinstance(agent, PatchAgent) or not agent.land:
                continue

            K = agent.original_food_value
            if K > reserve:
                base = 99 * K / reserve - 99
                if base > 0:
                    agent.growth_rate = base ** (-1 / self.regeneration_time)
                else:
                    agent.growth_rate = 0
            else:
                agent.growth_rate        = 0
                agent.original_food_value = 0

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self):
        import time
        current_step = self.schedule.steps
        t = {}

        # NetLogo go-volgorde: exploit-resources → burn-resources → regenerate → disaster
        # Stap 1: ALLE communities exploiteren (food, wood, clay)
        t0 = time.perf_counter()
        harvest_times = {}
        for agent in self.active_communities:
            ta = time.perf_counter()
            agent.exploit_resources()
            harvest_times[getattr(agent, 'site_name', agent.unique_id)] = (time.perf_counter() - ta) * 1000
        t["1_exploit_resources"] = time.perf_counter() - t0
        if harvest_times:
            slowest = sorted(harvest_times.items(), key=lambda x: -x[1])[:3]
            print(f"    exploit top-3: " + "  |  ".join(f"{n}: {t:.1f}ms" for n,t in slowest))

        # Stap 2: ALLE communities verbranden resources
        t0 = time.perf_counter()
        for agent in self.active_communities:
            agent.burn_resources()
        t["2_burn_resources"] = time.perf_counter() - t0

        # Stap 3: regeneratie (patches + workdays)
        t0 = time.perf_counter()
        self.regenerate()
        t["3_regenerate"] = time.perf_counter() - t0

        # Stap 4: rampen (brand + slechte oogst)
        t0 = time.perf_counter()
        self.disaster()
        t["4_disaster"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.datacollector.collect(self)
        t["5_datacollector"] = time.perf_counter() - t0

        total = sum(t.values())
        print(f"\n--- Stap {current_step} ({total*1000:.0f}ms totaal) ---")
        for name, t in sorted(t.items()):
            pct = t / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {name:<25} {t*1000:6.1f}ms  {pct:4.0f}%  {bar}")

        # NetLogo: periodisering-check staat NA go-body en NA tick
        # if ticks = 450 [ add-sites-ACH ]  /  if ticks = 650 [ add-sites-HELL ]
        if self.schedule.steps == 450:
            self._add_sites("ACH")
        if self.schedule.steps == 650:
            self._add_sites("HELL")

        self.schedule.steps += 1
        self.schedule.time  += 1

        if self.schedule.steps >= self.time_limit:
            self.running = False

    # =========================================================================
    # ADD SITES (periodisering)
    # =========================================================================

    def _add_sites(self, period):
        for agent in self.schedule.agents:
            if isinstance(agent, CommunityAgent) and agent.start_period == period:
                agent.active = True
                self.active_communities.append(agent)
                patch_here = self.patch_map.get(agent.pos)
                if patch_here:
                    patch_here.wood_maxStandingStock = 0
                    patch_here.food_fertility        = 0
                    patch_here.wood_regeneration     = False
                    patch_here.food_regeneration     = False
                    patch_here.clay_source           = False

    # =========================================================================
    # REGENERATE
    # =========================================================================

    def regenerate(self):
        import time
        t0 = time.perf_counter()

        # in NetLogo regenerate gebeurt in twee passes:
        #
        # Pass 1: ask patches [if food? = true and wood? = false [...]]
        #   -> enkel akkers: time_since_abandonment verhogen + voedsel regenereren
        #
        # Pass 2: ask patches [wood-updateStandingStock]
        #   -> ALLE land-patches: houtgroei + bos-terugkeer activeren
        #      (time_since_abandonment > forest_regrowth_lag -> wood? = true)

        for patch in self.land_patches:
            if patch.food_regeneration and not patch.wood_regeneration:
                # NetLogo: if food? = true and wood? = false
                patch.time_since_abandonment += 1
                patch.update_food_standing_stock()

        for patch in self.land_patches:
            # NetLogo: ask patches [wood-updateStandingStock] - geldt voor alle land-patches
            patch.update_wood_standing_stock()

        t1 = time.perf_counter()

        for agent in self.active_communities:
            debt_work = min(0, agent.workdays)
            debt_food = min(0, agent.food_workdays)
            agent.workdays      = agent.population * self.active_percentage / 100 * 365 + debt_work
            agent.food_workdays = agent.population * self.active_percentage / 100 * self.agriculture_days + debt_food
        t2 = time.perf_counter()

        self.bad_harvest_modifier = 1
        print(f"    regenerate: patch_loop={( t1-t0)*1000:.1f}ms  workday_reset={(t2-t1)*1000:.1f}ms  n_patches={len(self.land_patches)}")

    # =========================================================================
    # DISASTER
    # =========================================================================

    def disaster(self):
        fire_initial_patches = []
        drought = random.uniform(0, 1)

        for patch in self.land_patches:
            if patch.wood_standingStock > 0.72:
                p_base = 1 / (patch.fire_return_rate * 4)
                odds   = p_base / (1 - p_base) * drought
                if odds / (1 + odds) > random.uniform(0, 1):
                    patch.wood_standingStock = 0
                    patch.wood_age           = 0
                    patch.time_since_fire    = 0
                    fire_initial_patches.append(patch)

        for initial_patch in fire_initial_patches:
            u = max(random.uniform(0, 1), 1e-10)
            max_fire_size = -4 * np.log10(1 - u)
            if max_fire_size > 1:
                patches_burned = {initial_patch}
                while len(patches_burned) <= max_fire_size and random.uniform(0, 1) > 0.1:
                    peripheral = set()
                    for burned in patches_burned:
                        bx, by = burned.pos
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nb = self.patch_map.get((bx + dx, by + dy))
                            if nb and nb.wood_standingStock > 0.72 and nb not in patches_burned:
                                peripheral.add(nb)
                    if not peripheral:
                        break
                    newly = random.choice(list(peripheral))
                    newly.wood_standingStock = 0
                    newly.wood_age           = 0
                    newly.time_since_fire    = 0
                    patches_burned.add(newly)
                self.burn_size.append(len(patches_burned))

        if np.random.poisson(1 / (1 + self.bad_harvest_interval)) > 0:
            self.bad_harvest_modifier = 0.5

    # =========================================================================
    # RENDER CACHE UPDATE  (snelle sync voor visualisatie)
    # =========================================================================

    def _update_render_cache(self):
        """
        Werkt numpy render-arrays bij vanuit patch-objecten.
        Alleen land-patches — water verandert nooit.
        """
        H = self.grid.height
        ws  = self._rc_wood_stock
        ff  = self._rc_food_fert
        wr  = self._rc_wood_regen
        fr  = self._rc_food_regen
        fn  = self._rc_fire_now
        cq  = self._rc_clay_q
        for patch in self.land_patches:
            x, y = patch.pos
            row  = H - 1 - y
            ws[row, x] = patch.wood_standingStock
            ff[row, x] = patch.food_fertility
            wr[row, x] = patch.wood_regeneration
            fr[row, x] = patch.food_regeneration
            fn[row, x] = patch.time_since_fire == 0
            cq[row, x] = patch.land_is == "clay quarry"

    # =========================================================================
    # HULPFUNCTIES
    # =========================================================================

    def _get_patch(self, x, y):
        contents = self.grid.get_cell_list_contents([(x, y)])
        for a in contents:
            if isinstance(a, PatchAgent):
                return a
        return None