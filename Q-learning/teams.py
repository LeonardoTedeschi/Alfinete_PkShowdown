import numpy as np
import re
from poke_env.teambuilder import Teambuilder

class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.packed_teams = []
        for team in teams:
            try:
                parsed_team = self.parse_showdown_team(team)
                packed_team = self.join_team(parsed_team)
                self.packed_teams.append(packed_team)
            except Exception as e:
                print(f"Erro ao processar time: {e}")

    def yield_team(self):
        return np.random.choice(self.packed_teams)

_RAW_TEAMS_TEXT = """
=== [gen9nationaldex] Time-1 ===

Ledian @ Light Clay  
Ability: Early Bird  
Tera Type: Bug  
EVs: 252 HP / 252 SpD  
Careful Nature  
- Rest  
- Light Screen  
- Reflect  
- U-turn  

Kyurem @ Life Orb  
Ability: Pressure  
Tera Type: Dragon  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Dragon Dance  
- Dragon Claw  
- Icicle Spear  
- Iron Head  

Skorupi @ Eviolite  
Ability: Battle Armor  
Tera Type: Poison  
EVs: 252 HP / 252 Atk  
Adamant Nature  
- Taunt  
- Toxic Spikes  
- Knock Off  
- Poison Jab  

Regirock @ Leftovers  
Ability: Sturdy  
Tera Type: Rock  
EVs: 252 HP / 4 Atk / 252 Def  
Impish Nature  
- Body Press  
- Curse  
- Stealth Rock  
- Heavy Slam  

Oinkologne (M) @ Rocky Helmet  
Ability: Lingering Aroma  
Tera Type: Normal  
EVs: 248 HP / 252 Atk / 8 SpD  
Adamant Nature  
- Body Slam  
- High Horsepower  
- Sleep Talk  
- Rest  

Altaria @ Altarianite  
Ability: Natural Cure  
Tera Type: Dragon  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Defog  
- Heal Bell  
- Heat Wave  
- Moonblast  


=== [gen9nationaldex] Time-2 ===

Gliscor @ Toxic Orb  
Ability: Poison Heal  
Tera Type: Ground  
EVs: 252 HP / 4 Atk / 252 Spe  
Quiet Nature  
- Taunt  
- Stealth Rock  
- U-turn  
- Earthquake  

Volcarona @ Heavy-Duty Boots  
Ability: Swarm  
Tera Type: Bug  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Quiver Dance  
- Giga Drain  
- Bug Buzz  
- Flamethrower  

Samurott-Hisui @ Life Orb  
Ability: Sharpness  
Tera Type: Water  
EVs: 252 Atk / 4 SpA / 252 Spe  
Naive Nature  
- Aqua Cutter  
- Air Slash  
- Sacred Sword  
- Sucker Punch  

Cloyster @ Focus Sash  
Ability: Skill Link  
Tera Type: Water  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Shell Smash  
- Pin Missile  
- Rock Blast  
- Icicle Spear  

Sableye-Mega @ Sablenite  
Ability: Magic Bounce  
Happiness: 0  
Tera Type: Dark  
EVs: 248 HP / 8 Def / 252 SpD  
Calm Nature  
IVs: 0 Atk  
- Will-O-Wisp  
- Taunt  
- Recover  
- Foul Play  

Roserade @ Life Orb  
Ability: Technician  
Tera Type: Grass  
EVs: 4 HP / 252 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Sludge Bomb  
- Energy Ball  
- Dazzling Gleam  
- Extrasensory  


=== [gen9nationaldex] Time-4 ===

Da Kicker (Zapdos-Galar) @ Choice Band  
Ability: Defiant  
Tera Type: Flying  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Brave Bird  
- Close Combat  
- U-turn  
- Knock Off  

Blissey @ Heavy-Duty Boots  
Ability: Natural Cure  
Tera Type: Water  
EVs: 4 HP / 252 Def / 252 SpD  
Calm Nature  
IVs: 0 Atk  
- Soft-Boiled  
- Thunder Wave  
- Seismic Toss  
- Stealth Rock  

Gliscor @ Toxic Orb  
Ability: Poison Heal  
Tera Type: Normal  
EVs: 244 HP / 36 Def / 216 SpD / 12 Spe  
Jolly Nature  
- Swords Dance  
- Facade  
- Knock Off  
- Protect  

Skarmory @ Rocky Helmet  
Ability: Sturdy  
Tera Type: Dragon  
EVs: 252 HP / 160 Def / 96 Spe  
Bold Nature  
IVs: 0 Atk  
- Body Press  
- Whirlwind  
- Roost  
- Spikes  

Pecharunt @ Air Balloon  
Ability: Poison Puppeteer  
Tera Type: Ghost  
EVs: 252 HP / 4 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Malignant Chain  
- Shadow Ball  
- Parting Shot  
- Recover  

Absol-Mega @ Absolite  
Ability: Magic Bounce  
Tera Type: Dark  
EVs: 4 Atk / 252 SpA / 252 Spe  
Hasty Nature  
- Dark Pulse  
- Ice Beam  
- Iron Tail  
- Will-O-Wisp  


=== [gen9nationaldex] Time-5 ===

Slowking-Galar @ Waterium Z  
Ability: Regenerator  
Tera Type: Poison  
EVs: 252 HP / 252 Def / 4 SpD  
Bold Nature  
IVs: 0 Atk  
- Sludge Bomb  
- Scald  
- Calm Mind  
- Slack Off  

Corviknight @ Rocky Helmet  
Ability: Pressure  
Tera Type: Flying  
EVs: 252 HP / 172 Def / 84 SpD  
Impish Nature  
IVs: 0 Atk  
- Body Press  
- Iron Defense  
- Roost  
- Defog  

Clefable @ Sticky Barb  
Ability: Magic Guard  
Tera Type: Fairy  
EVs: 252 HP / 4 Def / 252 SpD  
Calm Nature  
IVs: 0 Atk  
- Moonblast  
- Trick  
- Soft-Boiled  
- Stealth Rock  

Latias-Mega (F) @ Latiasite  
Ability: Levitate  
Tera Type: Dragon  
EVs: 252 HP / 4 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Psyshock  
- Mystical Fire  
- Calm Mind  
- Recover  

Gastrodon @ Leftovers  
Ability: Storm Drain  
Tera Type: Water  
EVs: 252 HP / 4 Def / 252 SpD  
Sassy Nature  
- Scald  
- Earthquake  
- Toxic  
- Recover  

Weavile @ Choice Band  
Ability: Pressure  
Tera Type: Dark  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Triple Axel  
- Knock Off  
- Ice Shard  
- Pursuit  


=== [gen9nationaldex] Time-6 ===

Landorus-Therian (M) @ Rockium Z  
Ability: Intimidate  
Tera Type: Ground  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Earthquake  
- Stealth Rock  
- Stone Edge  
- Swords Dance  

Greninja @ Heavy-Duty Boots  
Ability: Protean  
Tera Type: Water  
EVs: 176 Atk / 80 SpA / 252 Spe  
Hasty Nature  
- Spikes  
- U-turn  
- Ice Beam  
- Gunk Shot  

Corviknight @ Leftovers  
Ability: Pressure  
Tera Type: Flying  
EVs: 248 HP / 100 Def / 148 SpD / 12 Spe  
Careful Nature  
- Substitute  
- Bulk Up  
- Brave Bird  
- Roost  

Tapu Fini @ Leftovers  
Ability: Misty Surge  
Tera Type: Water  
EVs: 248 HP / 8 Def / 12 SpA / 44 SpD / 196 Spe  
Calm Nature  
IVs: 0 Atk  
- Moonblast  
- Defog  
- Scald  
- Taunt  

Lopunny-Mega @ Lopunnite  
Ability: Scrappy  
Happiness: 0  
Tera Type: Normal  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Fake Out  
- U-turn  
- Close Combat  
- Frustration  

Magnezone @ Choice Specs  
Ability: Magnet Pull  
Tera Type: Electric  
EVs: 4 HP / 252 SpA / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Hidden Power [Fire]  
- Thunderbolt  
- Volt Switch  
- Flash Cannon  


=== [gen9nationaldex] Time-7 ===

Mawile-Mega @ Mawilite  
Ability: Intimidate  
Tera Type: Steel  
EVs: 92 HP / 252 Atk / 164 Spe  
Adamant Nature  
- Thunder Punch  
- Play Rough  
- Sucker Punch  
- Focus Punch  

Heatran @ Leftovers  
Ability: Flash Fire  
Tera Type: Fire  
EVs: 252 HP / 128 SpD / 128 Spe  
Calm Nature  
IVs: 0 Atk  
- Magma Storm  
- Stealth Rock  
- Toxic  
- Earth Power  

Tapu Fini @ Leftovers  
Ability: Misty Surge  
Tera Type: Water  
EVs: 252 HP / 80 Def / 20 SpA / 84 SpD / 72 Spe  
Calm Nature  
IVs: 0 Atk  
- Taunt  
- Defog  
- Scald  
- Moonblast  

Landorus-Therian (M) @ Flyinium Z  
Ability: Intimidate  
Tera Type: Ground  
EVs: 100 HP / 252 Atk / 156 Spe  
Adamant Nature  
- Fly  
- Swords Dance  
- Gravity  
- Earthquake  

Tangrowth @ Rocky Helmet  
Ability: Regenerator  
Tera Type: Grass  
EVs: 252 HP / 252 Def / 4 SpA  
Bold Nature  
IVs: 0 Atk  
- Sleep Powder  
- Giga Drain  
- Hidden Power [Ice]  
- Focus Blast  

Weavile @ Choice Band  
Ability: Pressure  
Tera Type: Dark  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Icicle Crash  
- Knock Off  
- Pursuit  
- Ice Shard  


=== [gen9nationaldex] Time-8 ===

Heracross-Mega @ Heracronite  
Ability: Moxie  
Tera Type: Bug  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Swords Dance  
- Close Combat  
- Rock Blast  
- Substitute  

Kartana @ Choice Scarf  
Ability: Beast Boost  
Tera Type: Grass  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Leaf Blade  
- Sacred Sword  
- Smart Strike  
- Knock Off  

Mantine @ Heavy-Duty Boots  
Ability: Water Absorb  
Tera Type: Water  
EVs: 248 HP / 36 Def / 220 SpD / 4 Spe  
Calm Nature  
- Roost  
- Scald  
- Defog  
- Toxic  

Tapu Koko @ Electrium Z  
Ability: Electric Surge  
Tera Type: Electric  
EVs: 252 Atk / 4 SpA / 252 Spe  
Naive Nature  
- Wild Charge  
- Hidden Power [Ice]  
- U-turn  
- Roost  

Jirachi @ Leftovers  
Ability: Serene Grace  
Tera Type: Steel  
EVs: 248 HP / 188 SpD / 72 Spe  
Careful Nature  
- Wish  
- Protect  
- Iron Head  
- U-turn  

Landorus-Therian (M) @ Rocky Helmet  
Ability: Intimidate  
Tera Type: Ground  
EVs: 252 HP / 92 Def / 12 SpA / 152 Spe  
Modest Nature  
- Stealth Rock  
- Hidden Power [Ice]  
- Smack Down  
- Earth Power  


=== [gen9nationaldex] Time-9 ===

Kartana @ Normalium Z  
Ability: Beast Boost  
Tera Type: Grass  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Swords Dance  
- Sacred Sword  
- Leaf Blade  
- Giga Impact  

Greninja-Ash (M) @ Choice Specs  
Ability: Battle Bond  
Tera Type: Water  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
- Dark Pulse  
- Spikes  
- Water Shuriken  
- Surf  

Heatran @ Leftovers  
Ability: Flash Fire  
Tera Type: Fire  
EVs: 248 HP / 40 SpA / 88 SpD / 132 Spe  
Bashful Nature  
- Stealth Rock  
- Heavy Slam  
- Magma Storm  
- Earth Power  

Landorus-Therian (M) @ Leftovers  
Ability: Intimidate  
Tera Type: Ground  
EVs: 252 HP / 92 Def / 12 SpA / 152 Spe  
Modest Nature  
IVs: 30 Atk  
- Defog  
- Hidden Power [Ice]  
- Smack Down  
- Earth Power  

Toxapex @ Black Sludge  
Ability: Regenerator  
Tera Type: Poison  
EVs: 252 HP / 92 Def / 164 SpD  
Careful Nature  
IVs: 0 Atk  
- Scald  
- Toxic  
- Haze  
- Recover  

Latios-Mega (M) @ Latiosite  
Ability: Levitate  
Tera Type: Dragon  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Psychic  
- Mystical Fire  
- Aura Sphere  
- Recover  


=== [gen9nationaldex] Time-10 ===

Goodra (M) @ Assault Vest  
Ability: Sap Sipper  
Tera Type: Dragon  
EVs: 248 HP / 252 SpA / 8 Spe  
Modest Nature  
IVs: 0 Atk  
- Dragon Pulse  
- Flamethrower  
- Sludge Wave  
- Thunderbolt  

Sylveon (M) @ Leftovers  
Ability: Pixilate  
Tera Type: Fairy  
EVs: 248 HP / 244 Def / 16 SpD  
Calm Nature  
IVs: 0 Atk  
- Hyper Voice  
- Mystical Fire  
- Protect  
- Wish  

Toxtricity (M) @ Throat Spray  
Ability: Punk Rock  
Tera Type: Electric  
EVs: 4 Atk / 252 SpA / 252 Spe  
Rash Nature  
- Overdrive  
- Boomburst  
- Shift Gear  
- Fire Punch  

Seismitoad (M) @ Leftovers  
Ability: Water Absorb  
Tera Type: Water  
EVs: 252 HP / 252 Def / 4 SpD  
Relaxed Nature  
- Stealth Rock  
- Scald  
- Earthquake  
- Toxic  

Corviknight (M) @ Leftovers  
Ability: Pressure  
Tera Type: Flying  
EVs: 248 HP / 80 SpD / 180 Spe  
Impish Nature  
- Defog  
- Brave Bird  
- Roost  
- U-turn  

Galvantula @ Focus Sash  
Ability: Compound Eyes  
Tera Type: Bug  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Sticky Web  
- Thunder Wave  
- Thunder  
- Energy Ball  


=== [gen9nationaldex] Time-11 ===

Togekiss @ Leftovers  
Ability: Serene Grace  
Tera Type: Fairy  
EVs: 248 HP / 8 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Air Slash  
- Nasty Plot  
- Substitute  
- Thunder Wave  

Galvantula @ Focus Sash  
Ability: Compound Eyes  
Tera Type: Bug  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Sticky Web  
- Thunder Wave  
- Thunder  
- Energy Ball  

Cloyster @ Leftovers  
Ability: Skill Link  
Tera Type: Water  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Icicle Spear  
- Rock Blast  
- Ice Shard  
- Shell Smash  

Sandaconda @ Focus Sash  
Ability: Sand Spit  
Tera Type: Ground  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Stealth Rock  
- Glare  
- Earthquake  
- Rock Tomb  

Excadrill @ Focus Sash  
Ability: Sand Rush  
Tera Type: Ground  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Iron Head  
- Rock Slide  
- Earthquake  
- Rapid Spin  

Cinccino @ Leftovers  
Ability: Skill Link  
Tera Type: Normal  
EVs: 252 Atk / 4 Def / 252 Spe  
Jolly Nature  
- Bullet Seed  
- Knock Off  
- Rock Blast  
- Tail Slap  


=== [gen9nationaldex] RainDance ===

Pelipper @ Damp Rock  
Ability: Drizzle  
Tera Type: Ground  
EVs: 248 HP / 8 Atk / 252 SpA  
Quiet Nature  
- Hurricane  
- Weather Ball  
- U-turn  
- Roost  

Ludicolo @ Icium Z  
Ability: Swift Swim  
Tera Type: Fighting  
EVs: 4 HP / 252 SpA / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Ice Beam  
- Energy Ball  
- Hydro Pump  
- Focus Blast  

Swampert-Mega @ Swampertite  
Ability: Swift Swim  
Tera Type: Water  
EVs: 4 HP / 252 Atk / 252 Spe  
Adamant Nature  
- Stone Edge  
- Waterfall  
- Earthquake  
- Ice Punch  

Politoed @ Damp Rock  
Ability: Drizzle  
Tera Type: Fighting  
EVs: 252 HP / 4 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Weather Ball  
- Ice Beam  
- Focus Blast  
- Earth Power  

Kingdra @ Choice Specs  
Ability: Swift Swim  
Tera Type: Steel  
EVs: 4 HP / 252 SpA / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Dragon Pulse  
- Flash Cannon  
- Hurricane  
- Surf  

Thundurus-Therian (M) @ Heavy-Duty Boots  
Ability: Volt Absorb  
Tera Type: Electric  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Defog  
- Weather Ball  
- Wildbolt Storm  
- Grass Knot  


=== [gen9nationaldex] Sunnyday ===

Charizard @ Charizardite Y  
Ability: Solar Power  
Tera Type: Fire  
EVs: 4 HP / 252 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Solar Beam  
- Air Slash  
- Heat Wave  
- Scorching Sands  

Torkoal @ Heat Rock  
Ability: Drought  
Tera Type: Fire  
EVs: 248 HP / 8 Atk / 252 Def  
Relaxed Nature  
IVs: 0 Spe  
- Stealth Rock  
- Toxic  
- Gyro Ball  
- Rapid Spin  

Lilligant-Hisui (F) @ Aguav Berry  
Ability: Chlorophyll  
Tera Type: Grass  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Close Combat  
- Leaf Blade  
- Ice Spinner  
- Acrobatics  

Volcarona @ Heavy-Duty Boots  
Ability: Swarm  
Tera Type: Bug  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Bug Buzz  
- Giga Drain  
- Quiver Dance  
- Heat Wave  

Great Tusk @ Icium Z  
Ability: Protosynthesis  
Tera Type: Ground  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Ice Spinner  
- Close Combat  
- Earthquake  
- Stone Edge  

Slowking @ Leftovers  
Ability: Regenerator  
Tera Type: Fairy  
EVs: 252 HP / 4 Def / 252 SpD  
Sassy Nature  
IVs: 0 Atk / 0 Spe  
- Scald  
- Future Sight  
- Slack Off  
- Teleport  


=== [gen9nationaldex] SandStream ===

Tyranitar @ Darkinium Z  
Ability: Sand Stream  
Tera Type: Rock  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Pursuit  
- Aqua Tail  
- Stone Edge  
- Superpower  

Hippowdon @ Smooth Rock  
Ability: Sand Stream  
Tera Type: Ground  
EVs: 252 HP / 4 Atk / 252 Def  
Impish Nature  
- Slack Off  
- Toxic  
- Stealth Rock  
- Earthquake  

Dracozolt @ Air Balloon  
Ability: Sand Rush  
Tera Type: Electric  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Bolt Beak  
- Iron Tail  
- Outrage  
- Fire Fang  

Garchomp @ Garchompite  
Ability: Rough Skin  
Tera Type: Dragon  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Stone Edge  
- Swords Dance  
- Earthquake  
- Iron Head  

Ferrothorn @ Rocky Helmet  
Ability: Iron Barbs  
Tera Type: Grass  
EVs: 248 HP / 8 Atk / 252 Def  
Impish Nature  
- Leech Seed  
- Toxic  
- Power Whip  
- Body Press  

Excadrill @ Life Orb  
Ability: Sand Rush  
Tera Type: Ground  
EVs: 4 HP / 252 Atk / 252 Spe  
- X-Scissor  
- Poison Jab  
- Iron Head  
- Shadow Claw  


=== [gen9nationaldex] Hail ===

Ninetales-Alola @ Icy Rock  
Ability: Snow Warning  
Tera Type: Ice  
EVs: 252 HP / 4 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Aurora Veil  
- Safeguard  
- Hypnosis  
- Freeze-Dry  

Sandslash-Alola @ Life Orb  
Ability: Slush Rush  
Tera Type: Ice  
EVs: 4 HP / 252 Atk / 252 Spe  
Jolly Nature  
- Aqua Tail  
- Iron Head  
- Icicle Crash  
- Leech Life  

Mandibuzz @ Leftovers  
Ability: Overcoat  
Tera Type: Dark  
EVs: 252 HP / 4 Atk / 252 Def  
Impish Nature  
- Roost  
- Foul Play  
- Knock Off  
- Toxic  

Diancie @ Diancite  
Ability: Magic Bounce  
Tera Type: Rock  
EVs: 4 Atk / 252 SpA / 252 Spe  
Naive Nature  
- Moonblast  
- Diamond Storm  
- Mystical Fire  
- Stealth Rock  

Togekiss @ Assault Vest  
Ability: Serene Grace  
Tera Type: Fairy  
EVs: 252 HP / 4 Def / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Air Slash  
- Tri Attack  
- Draining Kiss  
- Heal Bell  

Rotom-Wash @ Leftovers  
Ability: Levitate  
Tera Type: Electric  
EVs: 252 HP / 252 Def / 4 SpD  
Bold Nature  
- Volt Switch  
- Will-O-Wisp  
- Pain Split  
- Hydro Pump  


=== [gen9nationaldex] TrickRoom ===

Hatterene (F) @ Focus Sash  
Ability: Magic Bounce  
Tera Type: Psychic  
EVs: 252 HP / 252 SpA / 4 SpD  
Quiet Nature  
IVs: 0 Atk  
- Trick Room  
- Mystical Fire  
- Psychic  
- Draining Kiss  

Marowak-Alola @ Thick Club  
Ability: Rock Head  
Tera Type: Fire  
EVs: 248 HP / 252 Atk / 8 SpD  
Brave Nature  
- Flare Blitz  
- Stone Edge  
- Shadow Bone  
- Bonemerang  

Crawdaunt @ Life Orb  
Ability: Adaptability  
Tera Type: Water  
EVs: 252 HP / 252 Atk / 4 SpD  
Brave Nature  
IVs: 0 Spe  
- Aqua Jet  
- Close Combat  
- Crabhammer  
- Knock Off  

Porygon-Z @ Eviolite  
Ability: Adaptability  
Tera Type: Normal  
EVs: 252 HP / 252 SpA / 4 SpD  
Quiet Nature  
IVs: 0 Atk  
- Tri Attack  
- Trick Room  
- Teleport  
- Nasty Plot  

Abomasnow-Mega @ Abomasite  
Ability: Snow Warning  
Tera Type: Grass  
EVs: 248 HP / 252 SpA / 8 SpD  
Quiet Nature  
IVs: 0 Spe  
- Blizzard  
- Focus Blast  
- Shadow Ball  
- Giga Drain  

Stakataka @ Aguav Berry  
Ability: Beast Boost  
Tera Type: Rock  
EVs: 252 HP / 4 Atk / 252 Def  
Relaxed Nature  
IVs: 0 Spe  
- Trick Room  
- Stealth Rock  
- Gyro Ball  
- Body Press  
"""

TEAMS_LIST = [t.strip() for t in re.split(r'=== \[.*?\] .*? ===', _RAW_TEAMS_TEXT) if t.strip()]