
# Class for resources & functions for exploitation, regeneration, fire, ...

from mesa import Agent
import random
import numpy as np

# from école.BCS_netlogo.environment import SAGAscapeModel


class RangerAgent(Agent):
    """
    Tijdelijke agent die alleen tijdens setup wordt gebruikt om
    least-cost paden te berekenen (NetLogo: rangers breed).
    Na setup worden ze verwijderd.
    """
    def __init__(self, unique_id, model, claiming=None, walking_cost=1.0):
        super().__init__(unique_id, model)
        self.claiming = claiming       # community-ID waarvoor deze ranger werkt
        self.walkingCost = walking_cost  # walkingTime van de patch waar hij staat

    def step(self):
        pass


class PatchAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.land_is = None
        self.elevation = None
        self.wood_age = 0
        self.wood_maxStandingStock = 0
        self.wood_rico = None
        self.wood_power = None
        self.wood_regeneration = False   # wood? in NetLogo
        self.wood_standingStock = 0
        self.food_fertility = 0
        self.original_food_value = 0
        self.food_regeneration = False   # food? in NetLogo
        self.growth_rate = 0
        self.clay_source = False         # clay? in NetLogo
        self.clay_quantity = 0
        self.land = False                # land? in NetLogo
        self.walkingTime = 1.0
        self.in_range_of = []            # lijst van community unique_ids
        self.claimed_cost = []           # lijst van least-cost afstanden (zelfde volgorde)
        self.fire_return_rate = 0
        self.time_since_fire = 0
        self.time_since_abandonment = 0
        self.owner = None

    def step(self):
        """
        Patch-stap: alleen regeneratie.
        Brand en workday-regeneratie worden op modelniveau afgehandeld (disaster / regenerate).
        """
        if not self.land:
            return

        # Hout- en voedselregeneratie worden aangeroepen vanuit model.regenerate(),
        # net zoals in NetLogo waar regenerate apart van patch-step staat.
        # Hier bijhouden van abandonment voor patches zonder eigenaar.
        # (In NetLogo staat dit in regenerate, alleen voor food?=true and wood?=false)
        pass

    # -------------------------------------------------------------------------
    # WOOD REGENERATION  (NetLogo: to wood-updateStandingStock)
    # -------------------------------------------------------------------------

    def update_wood_standing_stock(self):
        """
        Past houtvoorraad en bosleeftijd aan.
        Correspondeert met NetLogo's 'to wood-updateStandingStock'.
        """
        forest_regrowth_lag = self.model.forest_regrowth_lag

        # Na verlatingslag kan bos teruggroeien
        if self.time_since_abandonment > forest_regrowth_lag:
            self.wood_regeneration = True  # wood? = true

        if self.wood_regeneration:
            if self.wood_standingStock < self.wood_maxStandingStock:
                # Verhulst-achtige groeifunctie: S(t) = A * exp(B * t)
                self.wood_standingStock = self.wood_rico * np.exp(self.wood_power * self.wood_age)

            self.wood_standingStock = min(self.wood_standingStock, self.wood_maxStandingStock)
            self.wood_age += 1
            self.time_since_fire += 1

    # -------------------------------------------------------------------------
    # FOOD REGENERATION  (NetLogo: regenerate — food deel)
    # -------------------------------------------------------------------------

    def update_food_standing_stock(self):
        """
        Werkt de voedselproductiviteit bij met de Verhulst-formule uit NetLogo.
        Wordt alleen aangeroepen als food?=True en wood?=False (via model.regenerate).
        Correspondeert met het food-gedeelte van NetLogo's 'to regenerate'.
        """
        food_max = self.original_food_value

        if food_max <= 0:
            return

        if self.food_fertility < food_max:
            if self.food_fertility > 0:
                # NetLogo Verhulst-formule:
                # food-fertility-regeneration = (1 - F/K) /
                #   (1/K + r / (F * (1 - r)))
                denom = (1 / food_max) + (self.growth_rate / (self.food_fertility * (1 - self.growth_rate)))
                if denom != 0:
                    food_fertility_regeneration = (1 - self.food_fertility / food_max) / denom
                    self.food_fertility += food_fertility_regeneration
            else:
                # Startwaarde voor regeneratie (regeneration-reserve)
                self.food_fertility = self.model.regeneration_reserve

            self.food_fertility = min(self.food_fertility, food_max)

    # -------------------------------------------------------------------------
    # FIRE (brand wordt op modelniveau gestart — zie environment.py disaster())
    # -------------------------------------------------------------------------

    def fire_spread(self):
        """Placeholder — brandspreiding zit in model.disaster()."""
        pass


def adapted_fertility(fertility):
    """
    Schaalt vruchtbaarheidswaarden naar max 3.5.
    Correspondeert met NetLogo's 'to-report adapted-fertility'.
    """
    if fertility == 0:
        return 0
    elif fertility > 3.5:
        return 3.5
    else:
        return fertility * 2.8 / 3.5