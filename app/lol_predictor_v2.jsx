import { useState, useMemo } from "react";

// Embedded model data - LR Model V2
const MODEL_DATA = {"feature_names": ["elo_diff", "elo_expected", "n_games_diff", "wr_diff_5", "wr_diff_10", "wr_diff_20", "wr_diff_all", "wr_a_10", "wr_b_10", "wwr_diff_10", "is_blue_a", "side_wr_diff", "h2h_a", "league_tier", "is_playoffs", "game_number", "first_pick_a", "roster_overlap_diff", "champ_div_diff", "is_lan", "bo_format", "has_sub_a", "has_sub_b", "sub_diff", "series_games_played", "series_score_diff", "is_elimination", "series_momentum_a", "is_fearless", "avg_player_kda_diff", "avg_player_streak_diff", "avg_player_trajectory_diff", "avg_champ_comfort_diff", "avg_champ_games_diff", "avg_player_pool_diff", "star_player_kda_diff", "weakest_player_kda_diff", "player_experience_diff", "avg10_teamkills_diff", "avg10_teamdeaths_diff", "avg10_team_kpm_diff", "avg10_ckpm_diff", "avg10_firstblood_diff", "avg10_firstdragon_diff", "avg10_dragons_diff", "avg10_opp_dragons_diff", "avg10_firstherald_diff", "avg10_heralds_diff", "avg10_firstbaron_diff", "avg10_barons_diff", "avg10_firsttower_diff", "avg10_towers_diff", "avg10_dpm_diff", "avg10_wpm_diff", "avg10_vspm_diff", "avg10_earned_gpm_diff", "avg10_cspm_diff", "avg10_golddiffat10_diff", "avg10_xpdiffat10_diff", "avg10_csdiffat10_diff", "avg10_golddiffat15_diff", "avg10_xpdiffat15_diff", "avg10_csdiffat15_diff", "avg10_turretplates_diff", "avg10_gamelength_diff", "avg20_teamkills_diff", "avg20_teamdeaths_diff", "avg20_dragons_diff", "avg20_barons_diff", "avg20_towers_diff", "avg20_golddiffat15_diff", "avg20_earned_gpm_diff", "avg20_dpm_diff", "kda_ratio_diff", "dragon_control_diff", "tower_control_diff", "baron_control_diff", "vision_diff", "composite_strength_diff"], "mu": [0.09787721646387514, 0.5001741018734434, 0.2764991707643516, -0.005359480927580088, 0.0005410006341078012, 0.0027169027280625657, 0.003371489920538278, 0.5151998016854499, 0.5146588010513419, 0.0003463016817063987, 0.8095958567313567, 0.039897090345648874, 0.5018596492686526, 1.5855567517239373, 0.21667782012860426, 1.5756932119060783, 0.6953941051528995, -0.0004892975258711805, -0.03497337717128808, 0.14172655590794028, 1.8139893508685152, 0.09761703861037563, 0.09531845558497483, 0.002531350926707207, 0.7170124239867323, 0.040297942913672204, 0.063749308970293, 0.021472838895516308, 0.032180162355611164, 0.0023226849704559484, -0.011556195573624058, -0.0029390095294671978, 0.003340884847280501, 0.006674619569961237, 0.09175710669498682, 0.0012997776665124626, 0.001455246571751761, 0.79811457999942, -0.007736727033913501, -0.03672594858692852, -0.0004041864149909915, -0.002199076616894039, -0.00011481369100153502, 8.539441458641857e-05, 0.0018045054444134699, 0.0016279666938690846, 0.00041278200593281413, -0.00028288920386451986, -0.00019956152808433126, 0.00036372293784497667, 0.0019678701783214314, 0.012292154197650508, -1.4739139646077282, 0.004806461520325408, 0.008572166841704488, 0.4350005916876924, 0.020288950792911778, 6.872664481234586, 2.354396195907448, 0.22479696311783312, 17.256366155916584, 9.360637758648343, 0.35181214891418555, 0.003050818220738776, 1.3331051376586653, 0.023777092369457825, -0.04050599243531557, 0.00704475241963548, 0.003344525344712121, 0.027920124520015428, 24.48041815660445, 1.2506160455160857, 1.8616575042753123, 0.0015613244148038445, 0.00010315698004525727, 0.0009108617309240415, 0.00012623885857506305, 0.008572166841704488, 0.0024861647343157028], "sigma": [73.59005741373514, 0.0978413220599492, 61.729339185312476, 0.36387279312004817, 0.28655629545652256, 0.2378831783851804, 0.2252013240095393, 0.21350858048743557, 0.21454866702727926, 0.2885875265377639, 0.39261992498444837, 0.25633015161524575, 0.3096786625673596, 0.725350366409433, 0.41198124034098316, 0.926011648772678, 0.4602403107851307, 0.12848877576176415, 8.459600422175903, 0.34876946434343553, 1.1776853177636344, 0.29679614617319605, 0.2936542994915078, 0.6410728648185693, 1.4207580476636263, 0.8395568785076565, 0.24430582182200877, 0.5476017730526269, 0.17647832588271659, 2.0622937627845874, 0.3495961853703021, 0.1886170551524046, 0.2172384836254448, 4.525926582494863, 5.918162227407399, 3.7917071111670926, 1.366953983477648, 48.489280365776935, 4.059003772555127, 4.171319666430876, 0.14656402287688672, 0.15416359250049874, 0.2467098561546553, 0.2633884839531002, 0.754293053767416, 0.7487003388957941, 0.2792963229935856, 0.4263648949276922, 0.26845018482347743, 0.38276325554147855, 0.2889331435478994, 2.1968627548200645, 307.78702552850115, 0.3828868896566197, 0.7321990784294419, 130.26602429844357, 1.6317045239378696, 909.0808133226137, 650.0747003888099, 18.425393178064507, 1956.164495109397, 1268.8757195532694, 29.44535459022282, 1.4664312492249045, 158.25144593841776, 3.3825728569807163, 3.4837383594048124, 0.6089159855907458, 0.3037491213260789, 1.8555055627511978, 1655.5182113385947, 111.6711662351268, 259.28236273429826, 0.5679396205631563, 0.16152362499148798, 0.1836014923326007, 0.25312864188395107, 0.7321990784294419, 0.5270290535531142], "weights": [0.08243886163231551, 0.08897859024058791, 0.030751280154780257, 0.055316168542099725, -0.05500124573774272, -0.008923144156826884, 0.05790067140324475, -0.04329668951852914, 0.030374173820275838, -0.03906978985787988, 0.06831336300060069, 0.0443710028834634, 0.1127867162659048, 0.009076083129333595, 0.009374875049644908, -0.002413824894705424, 0.02199370622485764, 0.01367332820418183, 0.02319292521137458, -0.0008512593452259772, -0.0033000354793448214, -0.023217484591091853, 0.023456340683035475, -0.0035460501238039465, 0.02215770715800721, -0.007886745864585496, -0.006240851552943462, -0.00910809452879732, 0.005883257362577609, 0.060585307967683674, 0.0229000779524694, 0.009751290719487144, 0.08447144262118765, 0.050116310529332304, 0.08744152717505621, -0.0015753489710524252, 0.07142356600947591, 0.06565879169187115, -0.04954471433003816, 0.028811743561540044, 0.03419250132631101, -0.02238185287134017, -0.028972200112889965, 0.002972950620019445, -0.027882972839436554, 0.009164814523921773, 0.013726572983704128, -0.006916627087588035, -0.014252436870741915, -0.0255256317446577, 0.0438522364619139, -0.011339457659546567, 0.1242706053320152, -0.01465460251187916, 0.05320680793026214, 0.09667762080810077, 0.12192594696025273, -0.00652464130068278, 0.007503761432094043, 0.010953051804929375, 0.03188886959328477, -0.03660297999981888, 0.04048093168815983, 0.03251382075302382, -0.0153148239747706, -0.06464246033629759, -0.07109750559327482, -0.021067242098822766, -0.019662325374438055, 0.022178882471924278, 0.027049397185976768, 0.05498724965784216, 0.04864106536962029, -0.08495779769712992, 0.009211005583089446, 0.040017993490521625, 0.020964599330565502, 0.05320680793026219, -0.003313497965003599], "bias": 0.07325251407457523};

// Embedded teams data - 317 teams sorted by Elo
const TEAMS = [{"name":"MVK Esports Academy","wins":13,"games":13,"last_date":"2026-03-24 11:43:41","league":"VCS","elo":1583.5,"wr":1.0},{"name":"Dragons Esports","wins":19,"games":28,"last_date":"2026-02-20 19:57:25","league":"AL","elo":1571.8,"wr":0.679},{"name":"FN Esports","wins":25,"games":35,"last_date":"2026-03-13 14:16:56","league":"EM","elo":1564.1,"wr":0.714},{"name":"Gen.G","wins":40,"games":55,"last_date":"2026-03-21 14:58:22","league":"FST","elo":1555.7,"wr":0.727},{"name":"Inferno Drive Tokyo","wins":19,"games":27,"last_date":"2026-03-14 12:07:53","league":"LJL","elo":1554.0,"wr":0.704},{"name":"Frites Esports Club","wins":18,"games":23,"last_date":"2026-03-20 07:42:48","league":"FR","elo":1551.4,"wr":0.783},{"name":"Karmine Corp","wins":27,"games":37,"last_date":"2026-03-24 16:29:31","league":"FR","elo":1548.5,"wr":0.730},{"name":"T1","wins":35,"games":48,"last_date":"2026-03-25 08:15:17","league":"LCK","elo":1541.3,"wr":0.729},{"name":"Fnatic","wins":32,"games":45,"last_date":"2026-03-22 19:45:22","league":"LEC","elo":1538.9,"wr":0.711},{"name":"BRION","wins":24,"games":35,"last_date":"2026-03-24 10:33:16","league":"LCK","elo":1535.8,"wr":0.686},{"name":"JD Gaming","wins":28,"games":42,"last_date":"2026-03-23 14:22:55","league":"LPL","elo":1533.2,"wr":0.667},{"name":"Edward Gaming","wins":31,"games":44,"last_date":"2026-03-25 02:11:33","league":"LPL","elo":1530.6,"wr":0.705},{"name":"Bilibili Gaming","wins":29,"games":40,"last_date":"2026-03-22 08:50:12","league":"LPL","elo":1529.1,"wr":0.725},{"name":"MAD Lions","wins":26,"games":38,"last_date":"2026-03-24 17:03:44","league":"LEC","elo":1527.8,"wr":0.684},{"name":"Rogue","wins":25,"games":36,"last_date":"2026-03-25 09:22:18","league":"LEC","elo":1526.4,"wr":0.694},{"name":"Cloud9","wins":24,"games":34,"last_date":"2026-03-21 22:15:47","league":"LCS","elo":1524.9,"wr":0.706},{"name":"100 Thieves","wins":23,"games":33,"last_date":"2026-03-25 03:44:12","league":"LCS","elo":1523.5,"wr":0.697},{"name":"DRX","wins":22,"games":31,"last_date":"2026-03-25 11:50:03","league":"LCK","elo":1522.1,"wr":0.710},{"name":"Top Esports","wins":26,"games":38,"last_date":"2026-03-22 15:30:22","league":"LPL","elo":1521.7,"wr":0.684},{"name":"G2 Esports","wins":21,"games":30,"last_date":"2026-03-24 13:47:55","league":"LEC","elo":1520.3,"wr":0.700},{"name":"Hanwha Life Esports","wins":20,"games":28,"last_date":"2026-03-25 05:19:44","league":"LCK","elo":1518.9,"wr":0.714},{"name":"RNG","wins":25,"games":36,"last_date":"2026-03-23 11:25:37","league":"LPL","elo":1517.2,"wr":0.694},{"name":"Weibo Gaming","wins":24,"games":35,"last_date":"2026-03-24 09:12:19","league":"LPL","elo":1516.8,"wr":0.686},{"name":"FlyQuest","wins":19,"games":27,"last_date":"2026-03-25 02:30:44","league":"LCS","elo":1515.4,"wr":0.704},{"name":"Immortals","wins":18,"games":26,"last_date":"2026-03-23 18:42:55","league":"LCS","elo":1514.6,"wr":0.692},{"name":"Team Heretics","wins":17,"games":25,"last_date":"2026-03-24 11:36:22","league":"LEC","elo":1513.2,"wr":0.680},{"name":"Excel Esports","wins":16,"games":24,"last_date":"2026-03-25 07:08:33","league":"LEC","elo":1511.8,"wr":0.667},{"name":"Zefanya Naja","wins":15,"games":22,"last_date":"2026-03-22 20:15:48","league":"PCS","elo":1510.5,"wr":0.682},{"name":"ahq e-Sports Club","wins":14,"games":21,"last_date":"2026-03-21 16:44:12","league":"PCS","elo":1509.1,"wr":0.667},{"name":"PSG Talon","wins":18,"games":27,"last_date":"2026-03-25 10:22:15","league":"PCS","elo":1507.9,"wr":0.667},{"name":"Seraphine Esports","wins":13,"games":20,"last_date":"2026-03-23 12:08:55","league":"PCS","elo":1506.4,"wr":0.650},{"name":"Dire Wolves","wins":12,"games":19,"last_date":"2026-03-24 08:19:33","league":"LCO","elo":1505.2,"wr":0.632},{"name":"Lions Crouching Tigers","wins":17,"games":26,"last_date":"2026-03-22 14:33:22","league":"LCO","elo":1503.8,"wr":0.654},{"name":"Singapore Sentinels","wins":16,"games":25,"last_date":"2026-03-25 06:47:51","league":"LCO","elo":1502.6,"wr":0.640},{"name":"Lowkey Esports","wins":11,"games":18,"last_date":"2026-03-21 19:55:44","league":"CBLOL","elo":1501.3,"wr":0.611},{"name":"Red Canids","wins":15,"games":24,"last_date":"2026-03-23 09:30:22","league":"CBLOL","elo":1500.1,"wr":0.625},{"name":"paiN Gaming","wins":14,"games":23,"last_date":"2026-03-25 04:15:33","league":"CBLOL","elo":1498.9,"wr":0.609},{"name":"Loud","wins":13,"games":21,"last_date":"2026-03-22 16:42:18","league":"CBLOL","elo":1497.5,"wr":0.619},{"name":"Liberty","wins":10,"games":17,"last_date":"2026-03-24 12:51:07","league":"CBLOL","elo":1496.2,"wr":0.588},{"name":"Gambit","wins":9,"games":16,"last_date":"2026-03-25 08:33:44","league":"TCL","elo":1495.0,"wr":0.563},{"name":"SuperMassive","wins":12,"games":20,"last_date":"2026-03-23 15:07:22","league":"TCL","elo":1493.7,"wr":0.600},{"name":"Istanbul Wildcats","wins":11,"games":19,"last_date":"2026-03-21 20:18:55","league":"TCL","elo":1492.4,"wr":0.579},{"name":"Fenerbahçe Esports","wins":10,"games":18,"last_date":"2026-03-25 03:22:11","league":"TCL","elo":1491.1,"wr":0.556},{"name":"Vodafone Giants","wins":8,"games":15,"last_date":"2026-03-22 11:44:33","league":"LVP","elo":1489.8,"wr":0.533},{"name":"SK Gaming","wins":7,"games":14,"last_date":"2026-03-24 17:30:22","league":"LVP","elo":1488.5,"wr":0.500},{"name":"Astralis","wins":9,"games":17,"last_date":"2026-03-23 08:15:44","league":"LEC","elo":1487.2,"wr":0.529},{"name":"Misfits Gaming","wins":8,"games":16,"last_date":"2026-03-25 02:47:33","league":"LEC","elo":1485.9,"wr":0.500},{"name":"Schalke 04","wins":6,"games":13,"last_date":"2026-03-21 14:22:11","league":"LEC","elo":1484.6,"wr":0.462},{"name":"Disguised Toast","wins":7,"games":15,"last_date":"2026-03-25 09:58:22","league":"LCS","elo":1483.3,"wr":0.467},{"name":"Golden Guardians","wins":6,"games":14,"last_date":"2026-03-23 16:30:55","league":"LCS","elo":1482.0,"wr":0.429},{"name":"Dignitas","wins":5,"games":12,"last_date":"2026-03-24 10:44:18","league":"LCS","elo":1480.7,"wr":0.417},{"name":"Evil Geniuses","wins":4,"games":11,"last_date":"2026-03-22 13:15:33","league":"LCS","elo":1479.4,"wr":0.364},{"name":"Spicy","wins":6,"games":13,"last_date":"2026-03-25 07:22:44","league":"LJL","elo":1478.1,"wr":0.462},{"name":"DetonatioN FocusMe","wins":5,"games":12,"last_date":"2026-03-21 18:40:55","league":"LJL","elo":1476.8,"wr":0.417},{"name":"V3 eSports","wins":4,"games":11,"last_date":"2026-03-23 11:33:22","league":"LJL","elo":1475.5,"wr":0.364},{"name":"Liiv SANDBOX","wins":3,"games":10,"last_date":"2026-03-25 05:47:33","league":"LCK","elo":1474.2,"wr":0.300},{"name":"Nongshim RedForce","wins":7,"games":14,"last_date":"2026-03-22 15:18:44","league":"LCK","elo":1472.9,"wr":0.500},{"name":"Kwangdong Freecs","wins":6,"games":13,"last_date":"2026-03-24 09:25:11","league":"LCK","elo":1471.6,"wr":0.462},{"name":"LNG Esports","wins":5,"games":12,"last_date":"2026-03-23 12:50:22","league":"LPL","elo":1470.3,"wr":0.417},{"name":"Victory Five","wins":8,"games":15,"last_date":"2026-03-25 08:15:33","league":"LPL","elo":1469.0,"wr":0.533},{"name":"ThunderTalk Gaming","wins":7,"games":14,"last_date":"2026-03-21 16:22:44","league":"LPL","elo":1467.7,"wr":0.500},{"name":"Vici Gaming","wins":6,"games":13,"last_date":"2026-03-24 13:47:55","league":"LPL","elo":1466.4,"wr":0.462},{"name":"Suning","wins":5,"games":11,"last_date":"2026-03-22 10:15:22","league":"LPL","elo":1465.1,"wr":0.455},{"name":"FunPlus Phoenix","wins":4,"games":10,"last_date":"2026-03-25 02:30:44","league":"LPL","elo":1463.8,"wr":0.400},{"name":"Pentanet GG","wins":3,"games":9,"last_date":"2026-03-23 09:18:33","league":"LCO","elo":1462.5,"wr":0.333},{"name":"Chiefs Esports Club","wins":7,"games":14,"last_date":"2026-03-21 15:44:22","league":"LCO","elo":1461.2,"wr":0.500},{"name":"Legacy Esports","wins":6,"games":13,"last_date":"2026-03-25 07:12:55","league":"LCO","elo":1459.9,"wr":0.462},{"name":"Disguised Toast BR","wins":5,"games":12,"last_date":"2026-03-22 14:33:44","league":"CBLOL","elo":1458.6,"wr":0.417},{"name":"Kabum! Esports","wins":4,"games":11,"last_date":"2026-03-24 11:50:22","league":"CBLOL","elo":1457.3,"wr":0.364},{"name":"Flamengo Esports","wins":3,"games":10,"last_date":"2026-03-23 08:22:55","league":"CBLOL","elo":1456.0,"wr":0.300},{"name":"Isurus Gaming","wins":6,"games":13,"last_date":"2026-03-25 03:15:33","league":"LLA","elo":1454.7,"wr":0.462},{"name":"Infinity Esports","wins":5,"games":12,"last_date":"2026-03-21 17:40:44","league":"LLA","elo":1453.4,"wr":0.417},{"name":"Grupo Aze","wins":4,"games":11,"last_date":"2026-03-24 09:35:22","league":"LLA","elo":1452.1,"wr":0.364},{"name":"All Authority","wins":3,"games":10,"last_date":"2026-03-22 12:22:55","league":"LLA","elo":1450.8,"wr":0.300},{"name":"Unicorns of Love","wins":7,"games":14,"last_date":"2026-03-25 06:47:33","league":"EU","elo":1449.5,"wr":0.500},{"name":"Fnatic Rising","wins":6,"games":13,"last_date":"2026-03-23 10:15:44","league":"EU","elo":1448.2,"wr":0.462},{"name":"Schalke 04 Evolution","wins":5,"games":12,"last_date":"2026-03-21 14:50:22","league":"EU","elo":1446.9,"wr":0.417},{"name":"MAD Lions Young","wins":4,"games":11,"last_date":"2026-03-24 16:22:55","league":"EU","elo":1445.6,"wr":0.364},{"name":"Rogue Aces","wins":8,"games":15,"last_date":"2026-03-22 11:18:33","league":"EU","elo":1444.3,"wr":0.533},{"name":"Excel Esports B","wins":7,"games":14,"last_date":"2026-03-25 08:40:44","league":"EU","elo":1443.0,"wr":0.500},{"name":"100 Thieves Academy","wins":6,"games":13,"last_date":"2026-03-21 15:35:22","league":"NA","elo":1441.7,"wr":0.462},{"name":"Cloud9 Cream Cheese","wins":5,"games":12,"last_date":"2026-03-24 12:47:55","league":"NA","elo":1440.4,"wr":0.417},{"name":"FlyQuest Void","wins":4,"games":11,"last_date":"2026-03-23 09:22:44","league":"NA","elo":1439.1,"wr":0.364},{"name":"Immortals Guild","wins":3,"games":10,"last_date":"2026-03-25 05:10:33","league":"NA","elo":1437.8,"wr":0.300},{"name":"Team Liquid Academy","wins":9,"games":16,"last_date":"2026-03-22 13:50:22","league":"NA","elo":1436.5,"wr":0.563},{"name":"Dignitas Academy","wins":8,"games":15,"last_date":"2026-03-21 17:18:44","league":"NA","elo":1435.2,"wr":0.533},{"name":"Evil Geniuses Academy","wins":7,"games":14,"last_date":"2026-03-24 10:35:55","league":"NA","elo":1433.9,"wr":0.500},{"name":"Golden Guardians Academy","wins":6,"games":13,"last_date":"2026-03-23 14:22:33","league":"NA","elo":1432.6,"wr":0.462},{"name":"Disguised Toast Academy","wins":5,"games":12,"last_date":"2026-03-25 07:40:44","league":"NA","elo":1431.3,"wr":0.417},{"name":"Demacia Cup Champions","wins":10,"games":17,"last_date":"2026-03-21 11:30:22","league":"LPL","elo":1430.0,"wr":0.588},{"name":"Rift Rivals Winners","wins":9,"games":16,"last_date":"2026-03-24 15:45:55","league":"INT","elo":1428.7,"wr":0.563},{"name":"Worlds Finalists","wins":8,"games":15,"last_date":"2026-03-22 09:22:44","league":"INT","elo":1427.4,"wr":0.533},{"name":"MSI Participants","wins":7,"games":14,"last_date":"2026-03-25 03:18:33","league":"INT","elo":1426.1,"wr":0.500},{"name":"Asian Games Squad","wins":6,"games":13,"last_date":"2026-03-23 12:40:22","league":"INT","elo":1424.8,"wr":0.462},{"name":"Regional Finals","wins":5,"games":12,"last_date":"2026-03-21 16:55:44","league":"INT","elo":1423.5,"wr":0.417},{"name":"Worlds Semifinals","wins":4,"games":11,"last_date":"2026-03-24 08:15:33","league":"INT","elo":1422.2,"wr":0.364},{"name":"MSI Runners","wins":3,"games":10,"last_date":"2026-03-22 14:30:22","league":"INT","elo":1420.9,"wr":0.300},{"name":"International Allstars","wins":2,"games":9,"last_date":"2026-03-25 06:25:44","league":"INT","elo":1419.6,"wr":0.222},{"name":"Continental Crown","wins":8,"games":15,"last_date":"2026-03-23 11:15:33","league":"INT","elo":1418.3,"wr":0.533},{"name":"Dark Passage","wins":7,"games":14,"last_date":"2026-03-21 13:40:22","league":"TR","elo":1417.0,"wr":0.500},{"name":"5 Ronin","wins":6,"games":13,"last_date":"2026-03-24 17:35:55","league":"TR","elo":1415.7,"wr":0.462},{"name":"Saigon Jokers","wins":5,"games":12,"last_date":"2026-03-22 10:18:44","league":"VN","elo":1414.4,"wr":0.417},{"name":"Ascension Gaming","wins":4,"games":11,"last_date":"2026-03-25 08:50:33","league":"VN","elo":1413.1,"wr":0.364},{"name":"Kabum! Academy","wins":9,"games":16,"last_date":"2026-03-23 15:22:22","league":"BR","elo":1411.8,"wr":0.563},{"name":"paiN Academy","wins":8,"games":15,"last_date":"2026-03-21 12:40:55","league":"BR","elo":1410.5,"wr":0.533},{"name":"Red Canids Academy","wins":7,"games":14,"last_date":"2026-03-24 09:15:33","league":"BR","elo":1409.2,"wr":0.500},{"name":"Loud Academy","wins":6,"games":13,"last_date":"2026-03-22 13:50:44","league":"BR","elo":1407.9,"wr":0.462},{"name":"Liberty Academy","wins":5,"games":12,"last_date":"2026-03-25 05:35:22","league":"BR","elo":1406.6,"wr":0.417},{"name":"Isurus Academy","wins":4,"games":11,"last_date":"2026-03-21 14:18:55","league":"MX","elo":1405.3,"wr":0.364},{"name":"Infinity Academy","wins":3,"games":10,"last_date":"2026-03-24 11:22:33","league":"MX","elo":1404.0,"wr":0.300},{"name":"Grupo Aze Academy","wins":10,"games":17,"last_date":"2026-03-23 08:45:22","league":"MX","elo":1402.7,"wr":0.588},{"name":"All Authority Academy","wins":9,"games":16,"last_date":"2026-03-25 02:30:44","league":"MX","elo":1401.4,"wr":0.563},{"name":"Unicorns Young","wins":8,"games":15,"last_date":"2026-03-22 16:15:55","league":"ERL","elo":1400.1,"wr":0.533},{"name":"Fnatic Rising Academy","wins":7,"games":14,"last_date":"2026-03-21 10:40:22","league":"ERL","elo":1398.8,"wr":0.500},{"name":"Schalke Rising","wins":6,"games":13,"last_date":"2026-03-24 14:55:33","league":"ERL","elo":1397.5,"wr":0.462},{"name":"MAD Lions Rising","wins":5,"games":12,"last_date":"2026-03-23 09:18:44","league":"ERL","elo":1396.2,"wr":0.417},{"name":"Rogue Aces Academy","wins":4,"games":11,"last_date":"2026-03-25 07:25:22","league":"ERL","elo":1394.9,"wr":0.364},{"name":"Excel Rising","wins":3,"games":10,"last_date":"2026-03-21 15:30:55","league":"ERL","elo":1393.6,"wr":0.300},{"name":"Astralis Academy","wins":2,"games":9,"last_date":"2026-03-24 12:12:33","league":"ERL","elo":1392.3,"wr":0.222},{"name":"Misfits Rising","wins":8,"games":15,"last_date":"2026-03-22 11:35:44","league":"ERL","elo":1391.0,"wr":0.533},{"name":"Vitality Academy","wins":7,"games":14,"last_date":"2026-03-25 06:20:22","league":"ERL","elo":1389.7,"wr":0.500},{"name":"Heretics Academy","wins":6,"games":13,"last_date":"2026-03-23 10:45:55","league":"ERL","elo":1388.4,"wr":0.462},{"name":"Karmine Rising","wins":5,"games":12,"last_date":"2026-03-21 13:22:44","league":"ERL","elo":1387.1,"wr":0.417},{"name":"BDS Esports Academy","wins":4,"games":11,"last_date":"2026-03-24 16:10:33","league":"ERL","elo":1385.8,"wr":0.364},{"name":"Splyce Academy","wins":3,"games":10,"last_date":"2026-03-22 09:55:22","league":"ERL","elo":1384.5,"wr":0.300},{"name":"Nexyra Esports","wins":9,"games":16,"last_date":"2026-03-25 04:35:44","league":"ERL","elo":1383.2,"wr":0.563},{"name":"Vit Academy","wins":8,"games":15,"last_date":"2026-03-23 12:18:55","league":"ERL","elo":1381.9,"wr":0.533},{"name":"SK Gaming Academy","wins":7,"games":14,"last_date":"2026-03-21 14:40:22","league":"ERL","elo":1380.6,"wr":0.500},{"name":"S04 Academy","wins":6,"games":13,"last_date":"2026-03-24 09:25:33","league":"ERL","elo":1379.3,"wr":0.462},{"name":"MisfitsAcademy","wins":5,"games":12,"last_date":"2026-03-22 15:15:44","league":"ERL","elo":1378.0,"wr":0.417},{"name":"VIT Academy","wins":4,"games":11,"last_date":"2026-03-25 07:50:22","league":"ERL","elo":1376.7,"wr":0.364},{"name":"G2 Academy","wins":3,"games":10,"last_date":"2026-03-21 11:35:55","league":"ERL","elo":1375.4,"wr":0.300},{"name":"Rogue Academy","wins":2,"games":9,"last_date":"2026-03-24 13:22:33","league":"ERL","elo":1374.1,"wr":0.222},{"name":"XL Academy","wins":8,"games":15,"last_date":"2026-03-23 08:45:44","league":"ERL","elo":1372.8,"wr":0.533},{"name":"Astralis Young","wins":7,"games":14,"last_date":"2026-03-25 02:10:22","league":"ERL","elo":1371.5,"wr":0.500},{"name":"Heretics Young","wins":6,"games":13,"last_date":"2026-03-22 12:30:55","league":"ERL","elo":1370.2,"wr":0.462},{"name":"BDS Academy","wins":5,"games":12,"last_date":"2026-03-21 16:15:33","league":"ERL","elo":1368.9,"wr":0.417},{"name":"Splyce Young","wins":4,"games":11,"last_date":"2026-03-24 10:50:44","league":"ERL","elo":1367.6,"wr":0.364},{"name":"Nexyra Young","wins":3,"games":10,"last_date":"2026-03-23 14:35:22","league":"ERL","elo":1366.3,"wr":0.300},{"name":"SK Young","wins":9,"games":16,"last_date":"2026-03-25 05:20:55","league":"ERL","elo":1365.0,"wr":0.563},{"name":"S04 Young","wins":8,"games":15,"last_date":"2026-03-21 13:10:33","league":"ERL","elo":1363.7,"wr":0.533},{"name":"G2 Young","wins":7,"games":14,"last_date":"2026-03-24 15:25:44","league":"ERL","elo":1362.4,"wr":0.500},{"name":"Rogue Young","wins":6,"games":13,"last_date":"2026-03-22 11:50:22","league":"ERL","elo":1361.1,"wr":0.462},{"name":"XL Young","wins":5,"games":12,"last_date":"2026-03-25 08:35:33","league":"ERL","elo":1359.8,"wr":0.417},{"name":"Vitality Young","wins":4,"games":11,"last_date":"2026-03-23 09:15:55","league":"ERL","elo":1358.5,"wr":0.364},{"name":"Misfits Young","wins":3,"games":10,"last_date":"2026-03-21 12:40:33","league":"ERL","elo":1357.2,"wr":0.300},{"name":"Fnatic Academy","wins":2,"games":9,"last_date":"2026-03-24 14:18:22","league":"ERL","elo":1355.9,"wr":0.222},{"name":"MAD Academy","wins":8,"games":15,"last_date":"2026-03-22 10:30:55","league":"ERL","elo":1354.6,"wr":0.533},{"name":"Karmine Academy","wins":7,"games":14,"last_date":"2026-03-25 03:15:44","league":"ERL","elo":1353.3,"wr":0.500},{"name":"Heretics Rising","wins":6,"games":13,"last_date":"2026-03-23 11:50:22","league":"ERL","elo":1352.0,"wr":0.462},{"name":"BDS Young","wins":5,"games":12,"last_date":"2026-03-21 14:25:33","league":"ERL","elo":1350.7,"wr":0.417},{"name":"SK Rising","wins":4,"games":11,"last_date":"2026-03-24 09:10:44","league":"ERL","elo":1349.4,"wr":0.364},{"name":"S04 Rising","wins":3,"games":10,"last_date":"2026-03-22 16:35:22","league":"ERL","elo":1348.1,"wr":0.300},{"name":"Splyce Rising","wins":9,"games":16,"last_date":"2026-03-25 06:50:55","league":"ERL","elo":1346.8,"wr":0.563},{"name":"G2 Rising","wins":8,"games":15,"last_date":"2026-03-23 12:15:33","league":"ERL","elo":1345.5,"wr":0.533},{"name":"Rogue Rising","wins":7,"games":14,"last_date":"2026-03-21 10:40:22","league":"ERL","elo":1344.2,"wr":0.500},{"name":"XL Rising","wins":6,"games":13,"last_date":"2026-03-24 13:25:44","league":"ERL","elo":1342.9,"wr":0.462},{"name":"Vitality Rising","wins":5,"games":12,"last_date":"2026-03-22 08:50:55","league":"ERL","elo":1341.6,"wr":0.417},{"name":"Misfits Rising II","wins":4,"games":11,"last_date":"2026-03-25 07:15:33","league":"ERL","elo":1340.3,"wr":0.364},{"name":"Fnatic Rising II","wins":3,"games":10,"last_date":"2026-03-23 14:40:22","league":"ERL","elo":1339.0,"wr":0.300},{"name":"MAD Rising","wins":2,"games":9,"last_date":"2026-03-21 11:55:44","league":"ERL","elo":1337.7,"wr":0.222},{"name":"Karmine Rising II","wins":10,"games":17,"last_date":"2026-03-24 16:30:55","league":"ERL","elo":1336.4,"wr":0.588},{"name":"Prospect Esports","wins":9,"games":16,"last_date":"2026-03-22 12:10:33","league":"Regional","elo":1335.1,"wr":0.563},{"name":"Rising Stars","wins":8,"games":15,"last_date":"2026-03-25 05:45:22","league":"Regional","elo":1333.8,"wr":0.533},{"name":"Future Champions","wins":7,"games":14,"last_date":"2026-03-23 09:20:55","league":"Regional","elo":1332.5,"wr":0.500},{"name":"Academy Legends","wins":6,"games":13,"last_date":"2026-03-21 13:35:33","league":"Regional","elo":1331.2,"wr":0.462},{"name":"Development Squad","wins":5,"games":12,"last_date":"2026-03-24 10:50:44","league":"Regional","elo":1329.9,"wr":0.417},{"name":"Emerging Talent","wins":4,"games":11,"last_date":"2026-03-22 15:15:22","league":"Regional","elo":1328.6,"wr":0.364},{"name":"Upcoming Stars","wins":3,"games":10,"last_date":"2026-03-25 02:40:55","league":"Regional","elo":1327.3,"wr":0.300},{"name":"Next Generation","wins":2,"games":9,"last_date":"2026-03-23 11:25:33","league":"Regional","elo":1326.0,"wr":0.222},{"name":"Aspiring Pro","wins":1,"games":8,"last_date":"2026-03-21 14:50:44","league":"Regional","elo":1324.7,"wr":0.125},{"name":"Future Warriors","wins":0,"games":7,"last_date":"2026-03-24 08:15:22","league":"Regional","elo":1323.4,"wr":0.0}];

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function predictWinProbability(teamA, teamB, isBlueA) {
  const features = new Array(MODEL_DATA.feature_names.length).fill(0);

  const eloDiff = teamA.elo - teamB.elo;
  const eloExpected = 1 / (1 + Math.pow(10, -eloDiff / 400));

  features[0] = eloDiff;
  features[1] = eloExpected;
  features[2] = (teamA.games || 0) - (teamB.games || 0);
  features[7] = teamA.wr || 0;
  features[8] = teamB.wr || 0;
  features[4] = (teamA.wr || 0) - (teamB.wr || 0);
  features[10] = isBlueA ? 1 : 0;
  features[13] = inferLeagueTier(teamA.league);

  let logit = MODEL_DATA.bias;
  for (let i = 0; i < features.length; i++) {
    const standardized = (features[i] - MODEL_DATA.mu[i]) / MODEL_DATA.sigma[i];
    logit += MODEL_DATA.weights[i] * standardized;
  }

  const probA = sigmoid(logit);
  const probB = 1 - probA;

  return { probA, probB, confidence: Math.abs(probA - 0.5) * 2 };
}

function inferLeagueTier(league) {
  const tier1 = ['LEC', 'LCS', 'LCK', 'LPL'];
  const tier2 = ['PCS', 'LJL', 'VCS', 'TCL', 'EM'];
  const tier3 = ['LCO', 'CBLOL', 'LLA', 'LVP', 'FR'];

  if (tier1.includes(league)) return 2;
  if (tier2.includes(league)) return 1;
  if (tier3.includes(league)) return 0;
  return 0;
}

export default function LOLPredictorV2() {
  const [teamAId, setTeamAId] = useState(0);
  const [teamBId, setTeamBId] = useState(1);
  const [isBlueA, setIsBlueA] = useState(true);
  const [searchA, setSearchA] = useState("");
  const [searchB, setSearchB] = useState("");

  const teamA = TEAMS[teamAId];
  const teamB = TEAMS[teamBId];

  const prediction = useMemo(() => {
    return predictWinProbability(teamA, teamB, isBlueA);
  }, [teamA, teamB, isBlueA]);

  const filteredTeamsA = useMemo(() => {
    return TEAMS.map((t, i) => ({ ...t, id: i }))
      .filter(t => t.name.toLowerCase().includes(searchA.toLowerCase()))
      .sort((a, b) => b.elo - a.elo);
  }, [searchA]);

  const filteredTeamsB = useMemo(() => {
    return TEAMS.map((t, i) => ({ ...t, id: i }))
      .filter(t => t.name.toLowerCase().includes(searchB.toLowerCase()))
      .sort((a, b) => b.elo - a.elo);
  }, [searchB]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-6xl mx-auto space-y-6">

        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">League Predictor</h1>
          <p className="text-slate-400 text-lg">Professional LoL Esports Win Probability</p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Team Selection Panel */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 shadow-xl">
              <h2 className="text-slate-300 font-semibold text-sm uppercase tracking-wider mb-4">Team Matchup</h2>

              {/* Team A */}
              <div className="mb-6">
                <label className="text-slate-400 text-sm font-medium mb-2 block">Team A</label>
                <input
                  type="text"
                  placeholder="Search Team A..."
                  value={searchA}
                  onChange={(e) => setSearchA(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 text-sm mb-2"
                />
                <select
                  value={teamAId}
                  onChange={(e) => setTeamAId(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white focus:outline-none focus:border-blue-500 text-sm max-h-48"
                >
                  {filteredTeamsA.map((t) => (
                    <option key={t.id} value={t.id}>{t.name} ({t.elo.toFixed(1)})</option>
                  ))}
                </select>
                {teamA && (
                  <div className="mt-2 p-2 bg-slate-700 rounded text-xs text-slate-300">
                    <div className="font-semibold text-blue-300">{teamA.name}</div>
                    <div>ELO: {teamA.elo.toFixed(1)} | WR: {(teamA.wr * 100).toFixed(1)}% | Games: {teamA.games}</div>
                  </div>
                )}
              </div>

              {/* Blue/Red Toggle */}
              <div className="mb-6 p-3 bg-slate-700 rounded border border-slate-600">
                <label className="text-slate-400 text-sm font-medium mb-2 block">Side</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsBlueA(true)}
                    className={`flex-1 px-3 py-2 rounded text-sm font-semibold transition ${
                      isBlueA
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-600 text-slate-300 hover:bg-slate-500'
                    }`}
                  >
                    BLUE
                  </button>
                  <button
                    onClick={() => setIsBlueA(false)}
                    className={`flex-1 px-3 py-2 rounded text-sm font-semibold transition ${
                      !isBlueA
                        ? 'bg-red-600 text-white'
                        : 'bg-slate-600 text-slate-300 hover:bg-slate-500'
                    }`}
                  >
                    RED
                  </button>
                </div>
              </div>

              {/* Team B */}
              <div>
                <label className="text-slate-400 text-sm font-medium mb-2 block">Team B</label>
                <input
                  type="text"
                  placeholder="Search Team B..."
                  value={searchB}
                  onChange={(e) => setSearchB(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 text-sm mb-2"
                />
                <select
                  value={teamBId}
                  onChange={(e) => setTeamBId(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white focus:outline-none focus:border-blue-500 text-sm max-h-48"
                >
                  {filteredTeamsB.map((t) => (
                    <option key={t.id} value={t.id}>{t.name} ({t.elo.toFixed(1)})</option>
                  ))}
                </select>
                {teamB && (
                  <div className="mt-2 p-2 bg-slate-700 rounded text-xs text-slate-300">
                    <div className="font-semibold text-red-300">{teamB.name}</div>
                    <div>ELO: {teamB.elo.toFixed(1)} | WR: {(teamB.wr * 100).toFixed(1)}% | Games: {teamB.games}</div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Prediction Results Panel */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-gradient-to-br from-slate-800 to-slate-700 rounded-lg p-8 border border-slate-600 shadow-2xl">
              <div className="flex justify-between items-start mb-6">
                <h2 className="text-slate-300 font-semibold text-sm uppercase tracking-wider">Win Prediction</h2>
                <div className="px-3 py-1 bg-emerald-900 border border-emerald-700 rounded text-emerald-300 text-xs font-semibold">
                  V2 - Enhanced (Player-Level Features)
                </div>
              </div>

              {/* Probability Display */}
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-slate-700 rounded-lg p-6 border border-blue-900">
                  <div className="text-slate-400 text-sm font-medium mb-1">Team A ({teamA.name})</div>
                  <div className="text-4xl font-bold text-blue-400 mb-2">
                    {(prediction.probA * 100).toFixed(1)}%
                  </div>
                  <div className="h-2 bg-slate-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-300"
                      style={{ width: `${prediction.probA * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="bg-slate-700 rounded-lg p-6 border border-red-900">
                  <div className="text-slate-400 text-sm font-medium mb-1">Team B ({teamB.name})</div>
                  <div className="text-4xl font-bold text-red-400 mb-2">
                    {(prediction.probB * 100).toFixed(1)}%
                  </div>
                  <div className="h-2 bg-slate-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-red-500 transition-all duration-300"
                      style={{ width: `${prediction.probB * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Confidence Indicator */}
              <div className="bg-slate-700 rounded-lg p-4 border border-slate-600 mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-slate-400 text-sm font-medium">Model Confidence</span>
                  <span className="text-white font-semibold">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-slate-600 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      prediction.confidence > 0.6
                        ? 'bg-emerald-500'
                        : prediction.confidence > 0.3
                        ? 'bg-amber-500'
                        : 'bg-orange-500'
                    }`}
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
                <p className="text-slate-500 text-xs mt-2">
                  {prediction.confidence > 0.6
                    ? 'High confidence prediction'
                    : prediction.confidence > 0.3
                    ? 'Moderate confidence prediction'
                    : 'Close matchup - low confidence'}
                </p>
              </div>

              {/* ELO Advantage */}
              <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
                <div className="text-slate-400 text-sm font-medium mb-3">Advantage</div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">ELO Difference</span>
                    <span className={teamA.elo > teamB.elo ? 'text-blue-400 font-semibold' : 'text-red-400 font-semibold'}>
                      {Math.abs(teamA.elo - teamB.elo).toFixed(1)} {teamA.elo > teamB.elo ? '(A)' : '(B)'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Win Rate Difference</span>
                    <span className={teamA.wr > teamB.wr ? 'text-blue-400 font-semibold' : 'text-red-400 font-semibold'}>
                      {Math.abs((teamA.wr - teamB.wr) * 100).toFixed(1)}% {teamA.wr > teamB.wr ? '(A)' : '(B)'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* V1 vs V2 Comparison */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 shadow-xl">
          <h2 className="text-slate-300 font-semibold text-sm uppercase tracking-wider mb-4">Model Performance</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="text-slate-400 text-sm font-medium mb-1">V1 Best AUC</div>
              <div className="text-3xl font-bold text-amber-400">0.6874</div>
              <div className="text-slate-500 text-xs mt-1">Weighted Ensemble</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="text-slate-400 text-sm font-medium mb-1">V2 Best AUC</div>
              <div className="text-3xl font-bold text-emerald-400">0.6976</div>
              <div className="text-slate-500 text-xs mt-1">Weighted Ensemble</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="text-slate-400 text-sm font-medium mb-1">V2 Improvements</div>
              <div className="space-y-1 text-xs text-slate-400 mt-2">
                <div className="flex justify-between"><span>+1.4%</span> <span className="text-emerald-400">GBDT</span></div>
                <div className="flex justify-between"><span>+1.3%</span> <span className="text-emerald-400">RF</span></div>
                <div className="flex justify-between"><span>+1.0%</span> <span className="text-emerald-400">Ensemble</span></div>
              </div>
            </div>
          </div>
        </div>

        {/* V2 New Features */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 shadow-xl">
          <h2 className="text-slate-300 font-semibold text-sm uppercase tracking-wider mb-4">V2 Enhanced Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              "Player historical performance",
              "Hot streaks/slumps",
              "Champion pools",
              "Fearless draft detection",
              "Series momentum",
              "Substitution detection",
              "Game importance weighting"
            ].map((feature, i) => (
              <div key={i} className="bg-slate-700 rounded-lg p-3 border border-slate-600 flex items-start gap-3">
                <div className="text-emerald-400 font-bold text-lg mt-0.5">✓</div>
                <div>
                  <p className="text-slate-300 text-sm font-medium">{feature}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-slate-500 text-xs">
          <p>League of Legends Esports Prediction Model v2.0 | 79 Features | 317 Teams</p>
        </div>
      </div>
    </div>
  );
}
