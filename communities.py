# Class for the communities
# 1 agent = 1 nederzetting (hamlet / village / town) — exact zoals NetLogo

from mesa import Agent
import random


class CommunityAgent(Agent):
    def __init__(self, unique_id, model, site_name=None, population=0,
                 elevation_here=0, settlement_type=None, start_period="IA"):
        super().__init__(unique_id, model)

        self.site_name       = site_name
        self.population      = population
        self.elevation_here  = float(elevation_here)
        self.settlement_type = settlement_type
        self.start_period    = start_period
        self.active          = True

        active_pct            = model.active_percentage
        agri_days             = model.agriculture_days
        grain_per_grain_yield = model.grain_per_grain_yield

        self.workdays         = population * active_pct / 100 * 365
        self.food_workdays    = population * active_pct / 100 * agri_days
        self.food_requirement = population * 365 * model.food_demand_pc / 1000
        self.wood_requirement = population * (365 * model.wood_demand_pc + 0.0661 * elevation_here) / 695
        self.clay_requirement = population * model.clay_demand_pc / 1000
        self.wood_for_clay    = 0
        self.grain_per_grain_factor = grain_per_grain_yield / (grain_per_grain_yield - 1)

        self.food_stock  = 0.0
        self.clay_stock  = 0.0
        self.wood_stock  = 0.0

        self.total_food_effort = 0.0
        self.total_wood_effort = 0.0
        self.total_clay_effort = 0.0

        self.cumulative_food_stock = 0.0
        self.cumulative_wood_stock = 0.0
        self.cumulative_clay_stock = 0.0
        self.saved_food_workdays   = 0.0
        self.saved_wood_workdays   = 0.0
        self.saved_clay_workdays   = 0.0

        # Gevuld door model._setup_least_cost_distances()
        self.candidate_patches = []
        # {id(patch): cost} — snelle O(1) lookup i.p.v. lineair zoeken in lijsten
        self.cost_cache = {}

    # =========================================================================
    # STEP  (wordt niet meer direct aangeroepen — model roept exploit + burn apart aan)
    # =========================================================================

    def step(self):
        # Backward-compat: wordt niet gebruikt door model.step() maar kan nog wel worden aangeroepen
        # self.exploit_resources()
        # self.burn_resources()
        pass


    def exploit_resources(self):
        """NetLogo: exploit-resources (food → wood → clay)"""
        self.food_harvest()
        self.wood_harvest()
        self.clay_harvest()

    # =========================================================================
    # FOOD HARVEST
    # =========================================================================

    def food_harvest(self):
        if not self.candidate_patches:
            return

        food_needed = (self.food_requirement
                       * (1 + random.uniform(0, 1))
                       * self.grain_per_grain_factor)

        # NetLogo: sort-on [(- food-fertility) / cost] candidate-patches
        # Elke tick opnieuw sorteren op actuele vruchtbaarheid/kost-ratio
        cache = self.cost_cache
        sorted_patches = sorted(
            [(p, cache.get(id(p), 1.0)) for p in self.candidate_patches
             if p.food_fertility > 0 and p.land],
            key=lambda x: x[0].food_fertility / max(0.001, x[1]),
            reverse=True
        )

        for target, effort in sorted_patches:
            if self.food_stock >= food_needed or self.food_workdays <= 0:
                break

            food_got  = target.food_fertility
            wood_got  = target.wood_standingStock

            target.food_fertility     = 0
            target.wood_standingStock = 0
            target.wood_age           = 0
            target.wood_regeneration  = False
            target.food_regeneration  = True
            target.time_since_abandonment = 0
            target.owner   = self.unique_id
            target.land_is = "cultivated"

            self.food_stock+= food_got
            self.cumulative_food_stock += food_got
            self.total_food_effort += effort

            #workdays
            # 42 persondays per ha per annum assumed. Add to this the amount of workdays
            # spent on migrating back and forth.(no.of trips * time back and forth per trip / hours of work per day(assumed 10))
            wd = 42 + 42 * 2 * effort / 10
            self.food_workdays -= wd
            self.workdays -= wd
            self.saved_food_workdays += wd

            if wood_got > 0:
                self.wood_stock            += wood_got
                self.cumulative_wood_stock += wood_got
                self.total_wood_effort     += effort
                hl    = max(random.gauss(29.21, 14.14), 4.5) / 695
                trips = wood_got / hl - 2
                if trips > 0:
                    wdd = trips * (2 * effort + hl * 49) / 10
                    self.workdays            -= wdd
                    self.saved_wood_workdays += wdd

        self.food_stock *= self.model.bad_harvest_modifier

    # =========================================================================
    # WOOD HARVEST
    # =========================================================================

    def wood_harvest(self):
        if not self.candidate_patches:
            return
        hl = max(random.gauss(29.21, 14.14), 4.5) / 695

        # NetLogo: sort-on [(- wood-standingStock) / cost] candidate-patches
        # Elke tick opnieuw sorteren op actuele houtstock/kost-ratio
        cache = self.cost_cache
        sorted_patches = sorted(
            [(p, cache.get(id(p), 1.0)) for p in self.candidate_patches
             if p.wood_standingStock > 0 and p.land],
            key=lambda x: x[0].wood_standingStock / max(0.001, x[1]),
            reverse=True
        )

        for target, effort in sorted_patches:
            if self.wood_stock >= self.wood_requirement or self.workdays <= 0:
                break

            wood_got = target.wood_standingStock
            target.wood_standingStock = 0
            target.wood_age           = 0

            self.wood_stock            += wood_got
            self.cumulative_wood_stock += wood_got
            self.total_wood_effort     += effort

            wd = (wood_got / hl) * (2 * effort + hl * 49) / 10
            self.workdays            -= wd
            self.saved_wood_workdays += wd

    # =========================================================================
    # CLAY HARVEST
    # =========================================================================

    def clay_harvest(self):
        if not self.candidate_patches:
            return

        # NetLogo: sort-on [(- clay-quantity) / cost] candidate-patches with [clay? = true]
        # Elke tick opnieuw sorteren op actuele klei/kost-ratio
        cache = self.cost_cache
        sorted_patches = sorted(
            [(p, cache.get(id(p), 1.0)) for p in self.candidate_patches
             if p.clay_source and p.clay_quantity > 0],
            key=lambda x: x[0].clay_quantity / max(0.001, x[1]),
            reverse=True
        )

        self.wood_for_clay = 0

        for target, effort in sorted_patches:
            if self.clay_stock >= self.clay_requirement or self.workdays <= 0:
                break

            wood_got = target.wood_standingStock
            clay_got = 19

            target.clay_quantity      -= clay_got
            if target.clay_quantity < self.model.clay_threshold * 10000 * 2:
                target.clay_source = False
            target.food_fertility     = 0
            target.wood_standingStock = 0
            target.wood_age           = 0
            target.wood_regeneration  = False
            target.food_regeneration  = False
            target.land_is            = "clay quarry"

            self.clay_stock            += clay_got
            self.cumulative_clay_stock += clay_got
            self.total_clay_effort     += effort
            self.wood_for_clay         += clay_got * self.model.kgs_wood_per_kg_clay * 1000 / 695

            wd = (0.193 * clay_got / 1.9
                  + (clay_got / 0.05) * effort * 2 * 6.5 / 10
                  + 4.5 / 0.980 * clay_got)
            self.workdays            -= wd
            self.saved_clay_workdays += wd

            if wood_got > 0:
                self.wood_stock            += wood_got
                self.cumulative_wood_stock += wood_got
                self.total_wood_effort     += effort
                hl    = max(random.gauss(29.21, 14.14), 4.5) / 695
                trips = wood_got / hl - 2
                if trips > 0:
                    wdd = trips * (2 * effort + hl * 49) / 10
                    self.workdays            -= wdd
                    self.saved_wood_workdays += wdd

    # =========================================================================
    # BURN RESOURCES
    # =========================================================================

    def burn_resources(self):
        # NetLogo: stocks kunnen onbeperkt negatief worden (schuld naar volgend jaar)
        self.food_stock = self.food_stock / self.grain_per_grain_factor - self.food_requirement
        self.wood_stock = self.wood_stock - self.wood_requirement - self.wood_for_clay
        self.clay_stock = self.clay_stock - self.clay_requirement