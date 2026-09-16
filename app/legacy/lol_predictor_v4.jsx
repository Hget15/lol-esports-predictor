import { useState, useMemo } from "react";

const T = {
"9Gaming Esports":{elo:36.1,wr:0.715,r:{top:{n:"Nanaue",kda:3.58,dpm:867,g:16,wr:62,cs:9.1,dp:26.4,gp:22.7,pool:9,ch:["Gwen","Gnar","Renekton"]},jng:{n:"Umi",kda:2.9,dpm:622,g:5,wr:40,cs:7.0,dp:21.5,gp:23.6,pool:5,ch:["Jayce","Qiyana","Viego"]},mid:{n:"BayMaxxx",kda:3.06,dpm:660,g:16,wr:62,cs:8.4,dp:20.6,gp:19.0,pool:9,ch:["Ryze","Taliyah","Akali"]},bot:{n:"Nuna",kda:4.37,dpm:821,g:16,wr:62,cs:9.6,dp:25.4,gp:25.9,pool:10,ch:["Ashe","Aphelios","Ezreal"]},sup:{n:"Dara2",kda:2.87,dpm:271,g:13,wr:62,cs:1.1,dp:8.5,gp:10.9,pool:9,ch:["Neeko","Braum","Seraphine"]}}},
"BNK FEARX":{elo:-48.9,wr:0.55,r:{top:{n:"Clear",kda:2.17,dpm:680,g:20,wr:45,cs:8.4,dp:23.2,gp:20.5,pool:10,ch:["Renekton","Gnar","Sion"]},jng:{n:"Raptor",kda:2.65,dpm:546,g:20,wr:45,cs:6.4,dp:19.0,gp:19.8,pool:10,ch:["Nocturne","Vi","Jarvan IV"]},mid:{n:"VicLa",kda:2.51,dpm:718,g:20,wr:45,cs:8.9,dp:24.6,gp:21.7,pool:12,ch:["Syndra","Taliyah","Ahri"]},bot:{n:"Diable",kda:2.52,dpm:735,g:20,wr:45,cs:10.0,dp:24.9,gp:27.2,pool:10,ch:["Yunara","Varus","Kai'Sa"]},sup:{n:"Kellin",kda:3.26,dpm:242,g:20,wr:45,cs:1.0,dp:8.4,gp:10.8,pool:10,ch:["Karma","Rakan","Renata Glasc"]}}},
"Bushido Wildcats":{elo:27.6,wr:0.59,r:{top:{n:"StarScreen",kda:2.3,dpm:718,g:20,wr:45,cs:8.0,dp:24.9,gp:19.9,pool:10,ch:["Rumble","Gnar","K'Sante"]},jng:{n:"Diamondprox",kda:2.4,dpm:557,g:4,wr:50,cs:6.1,dp:17.0,gp:18.1,pool:2,ch:["Xin Zhao","Pantheon"]},mid:{n:"alix",kda:2.88,dpm:733,g:20,wr:45,cs:9.1,dp:25.2,gp:24.0,pool:11,ch:["Taliyah","Ryze","Orianna"]},bot:{n:"Kenal",kda:3.12,dpm:801,g:20,wr:45,cs:9.1,dp:27.6,gp:27.0,pool:11,ch:["Varus","Caitlyn","Corki"]},sup:{n:"Lekcyc",kda:3.35,dpm:205,g:20,wr:45,cs:1.2,dp:6.9,gp:10.2,pool:8,ch:["Alistar","Rakan","Bard"]}}},
"CTBC Flying Oyster":{elo:-30.2,wr:0.39,r:{top:{n:"Rest",kda:3.71,dpm:691,g:20,wr:45,cs:8.3,dp:23.0,gp:19.3,pool:9,ch:["Rumble","Sion","Renekton"]},jng:{n:"Shad0w",kda:3.97,dpm:574,g:20,wr:45,cs:7.5,dp:18.3,gp:21.3,pool:10,ch:["Wukong","Pantheon","Jarvan IV"]},mid:{n:"Pungyeon",kda:3.46,dpm:708,g:20,wr:45,cs:9.4,dp:22.9,gp:20.3,pool:11,ch:["Taliyah","Azir","Akali"]},bot:{n:"Doggo",kda:3.59,dpm:840,g:20,wr:45,cs:9.9,dp:27.2,gp:29.2,pool:10,ch:["Kai'Sa","Lucian","Yunara"]},sup:{n:"2274",kda:2.74,dpm:268,g:20,wr:45,cs:0.8,dp:8.6,gp:9.8,pool:12,ch:["Neeko","Nami","Rell"]}}},
"Clocks":{elo:8.4,wr:0.66,r:{top:{n:"Gecko",kda:4.71,dpm:739,g:20,wr:65,cs:7.8,dp:24.3,gp:22.2,pool:9,ch:["Jax","Rumble","Gwen"]},jng:{n:"Kangkuk",kda:5.53,dpm:632,g:20,wr:65,cs:7.0,dp:21.2,gp:22.3,pool:9,ch:["Xin Zhao","Lee Sin","Nidalee"]},mid:{n:"Razer",kda:3.15,dpm:598,g:20,wr:65,cs:8.0,dp:20.7,gp:19.0,pool:8,ch:["Galio","Sion","Orianna"]},bot:{n:"Fluid",kda:4.71,dpm:705,g:12,wr:67,cs:8.5,dp:22.6,gp:25.8,pool:7,ch:["Varus","Miss Fortune","Yunara"]},sup:{n:"chico",kda:3.64,dpm:330,g:20,wr:65,cs:1.1,dp:11.0,gp:10.9,pool:7,ch:["Neeko","Nautilus","Lulu"]}}},
"Cloud9":{elo:61.7,wr:0.64,r:{top:{n:"Thanatos",kda:2.28,dpm:600,g:20,wr:45,cs:9.0,dp:23.0,gp:21.2,pool:9,ch:["Renekton","Ambessa","Rumble"]},jng:{n:"Blaber",kda:2.64,dpm:377,g:20,wr:45,cs:7.3,dp:14.2,gp:20.4,pool:11,ch:["Jarvan IV","Wukong","Sejuani"]},mid:{n:"APA",kda:2.51,dpm:665,g:20,wr:45,cs:9.0,dp:25.3,gp:21.3,pool:10,ch:["Ryze","Taliyah","Annie"]},bot:{n:"Zven",kda:2.33,dpm:759,g:20,wr:45,cs:10.1,dp:29.2,gp:27.3,pool:9,ch:["Corki","Aphelios","Sivir"]},sup:{n:"Vulcan",kda:3.85,dpm:218,g:20,wr:45,cs:1.1,dp:8.3,gp:9.8,pool:10,ch:["Nami","Neeko","Karma"]}}},
"DN SOOPers":{elo:-41.4,wr:0.47,r:{top:{n:"DuDu",kda:3.92,dpm:735,g:20,wr:55,cs:9.1,dp:26.3,gp:22.2,pool:11,ch:["Gwen","Rek'Sai","Kennen"]},jng:{n:"Pyosik",kda:5.77,dpm:469,g:20,wr:55,cs:7.2,dp:16.1,gp:20.3,pool:12,ch:["Wukong","Naafiri","Lee Sin"]},mid:{n:"Clozer",kda:3.47,dpm:670,g:20,wr:55,cs:9.4,dp:23.2,gp:20.5,pool:13,ch:["Viktor","Orianna","Ahri"]},bot:{n:"deokdam",kda:3.48,dpm:765,g:20,wr:55,cs:9.3,dp:26.2,gp:26.5,pool:10,ch:["Yunara","Corki","Varus"]},sup:{n:"Peter",kda:2.9,dpm:216,g:20,wr:50,cs:1.1,dp:7.8,gp:10.4,pool:7,ch:["Rakan","Neeko","Rell"]}}},
"Deep Cross Gaming":{elo:-2.2,wr:0.61,r:{top:{n:"Flauren",kda:3.2,dpm:692,g:20,wr:50,cs:8.5,dp:23.7,gp:19.1,pool:13,ch:["Sion","Renekton","Gwen"]},jng:{n:"POP9",kda:3.26,dpm:472,g:20,wr:50,cs:7.0,dp:16.3,gp:19.6,pool:10,ch:["Dr. Mundo","Xin Zhao","Jarvan IV"]},mid:{n:"HongSuo",kda:3.58,dpm:701,g:20,wr:50,cs:9.0,dp:24.5,gp:21.8,pool:12,ch:["Azir","Taliyah","Ahri"]},bot:{n:"Feng",kda:3.39,dpm:803,g:20,wr:50,cs:9.7,dp:27.9,gp:28.5,pool:13,ch:["Varus","Sivir","Yunara"]},sup:{n:"ShiauC",kda:2.8,dpm:212,g:20,wr:50,cs:1.2,dp:7.6,gp:11.0,pool:13,ch:["Rell","Nautilus","Poppy"]}}},
"Disguised":{elo:0.5,wr:0.56,r:{top:{n:"Castle",kda:2.1,dpm:627,g:17,wr:41,cs:8.4,dp:24.5,gp:19.8,pool:9,ch:["Renekton","Jax","Galio"]},jng:{n:"KryRa",kda:3.14,dpm:399,g:11,wr:36,cs:7.1,dp:15.8,gp:19.5,pool:8,ch:["Wukong","Dr. Mundo","Pantheon"]},mid:{n:"Callme",kda:2.71,dpm:636,g:17,wr:41,cs:9.6,dp:25.5,gp:22.9,pool:11,ch:["Ryze","Aurora","Yone"]},bot:{n:"sajed",kda:3.46,dpm:654,g:17,wr:41,cs:9.8,dp:25.5,gp:28.0,pool:8,ch:["Kai'Sa","Corki","Lucian"]},sup:{n:"Lyonz",kda:3.95,dpm:228,g:17,wr:41,cs:1.0,dp:9.2,gp:9.7,pool:9,ch:["Bard","Nami","Nautilus"]}}},
"Dplus Kia":{elo:-25.8,wr:0.5,r:{top:{n:"Siwoo",kda:2.05,dpm:657,g:20,wr:50,cs:8.9,dp:22.7,gp:20.1,pool:9,ch:["Sion","Jax","Gwen"]},jng:{n:"Lucid",kda:2.37,dpm:507,g:20,wr:50,cs:6.6,dp:17.5,gp:19.7,pool:11,ch:["Wukong","Aatrox","Lee Sin"]},mid:{n:"ShowMaker",kda:2.39,dpm:661,g:20,wr:50,cs:8.8,dp:22.7,gp:21.2,pool:14,ch:["Ahri","Azir","Cassiopeia"]},bot:{n:"Smash",kda:3.58,dpm:823,g:20,wr:50,cs:10.7,dp:28.9,gp:29.0,pool:12,ch:["Ashe","Ezreal","Ziggs"]},sup:{n:"Career",kda:2.92,dpm:228,g:20,wr:50,cs:1.1,dp:8.1,gp:9.9,pool:12,ch:["Seraphine","Nami","Alistar"]}}},
"FURIA":{elo:23.3,wr:0.68,r:{top:{n:"Guigo",kda:3.63,dpm:629,g:20,wr:75,cs:8.2,dp:22.9,gp:19.4,pool:11,ch:["K'Sante","Ornn","Sion"]},jng:{n:"Tatu",kda:7.05,dpm:563,g:20,wr:75,cs:7.5,dp:20.7,gp:22.4,pool:11,ch:["Xin Zhao","Pantheon","Wukong"]},mid:{n:"Tutsz",kda:4.66,dpm:625,g:20,wr:75,cs:8.9,dp:22.7,gp:21.4,pool:12,ch:["Akali","Ahri","Orianna"]},bot:{n:"Ayu",kda:5.45,dpm:672,g:20,wr:75,cs:9.6,dp:24.4,gp:26.1,pool:9,ch:["Yunara","Ashe","Corki"]},sup:{n:"JoJo",kda:8.63,dpm:254,g:20,wr:75,cs:1.2,dp:9.3,gp:10.8,pool:9,ch:["Seraphine","Rakan","Nami"]}}},
"Fast8":{elo:-9.7,wr:0.515,r:{top:{n:"Hannah",kda:1.08,dpm:427,g:5,wr:0,cs:7.0,dp:15.8,gp:16.0,pool:3,ch:["Sion","Ornn","K'Sante"]},jng:{n:"ice1",kda:1.85,dpm:540,g:5,wr:0,cs:8.0,dp:19.7,gp:26.4,pool:4,ch:["Zed","Lee Sin","Graves"]},mid:{n:"Karaage",kda:2.54,dpm:759,g:19,wr:42,cs:8.7,dp:25.5,gp:21.1,pool:9,ch:["Ryze","Ahri","Viktor"]},bot:{n:"MayR",kda:2.65,dpm:702,g:19,wr:42,cs:9.0,dp:23.9,gp:24.7,pool:8,ch:["Ezreal","Jhin","Varus"]},sup:{n:"raku",kda:1.62,dpm:267,g:5,wr:0,cs:1.2,dp:9.8,gp:10.9,pool:4,ch:["Karma","Renata Glasc","Braum"]}}},
"French Flair":{elo:56.7,wr:0.75,r:{top:{n:"Adam",kda:3.46,dpm:765,g:20,wr:70,cs:8.9,dp:24.5,gp:21.0,pool:12,ch:["Sion","Volibear","Rumble"]},jng:{n:"NattyNatt",kda:3.5,dpm:525,g:20,wr:70,cs:7.7,dp:16.3,gp:20.2,pool:10,ch:["Pantheon","Jarvan IV","Aatrox"]},mid:{n:"SAKEN",kda:4.92,dpm:784,g:20,wr:70,cs:9.2,dp:25.4,gp:21.8,pool:12,ch:["Ryze","Aurora","LeBlanc"]},bot:{n:"3XA",kda:4.72,dpm:799,g:20,wr:70,cs:10.4,dp:25.2,gp:26.6,pool:8,ch:["Corki","Varus","Ashe"]},sup:{n:"Targamas",kda:5.31,dpm:267,g:20,wr:70,cs:1.1,dp:8.5,gp:10.4,pool:10,ch:["Nami","Nautilus","Neeko"]}}},
"Fukuoka SoftBank HAWKS gaming":{elo:-32.5,wr:0.51,r:{top:{n:"Evi",kda:2.6,dpm:677,g:20,wr:55,cs:8.3,dp:26.7,gp:19.2,pool:8,ch:["Sion","Zaahen","Rumble"]},jng:{n:"Van",kda:3.53,dpm:387,g:20,wr:55,cs:6.6,dp:15.1,gp:19.9,pool:11,ch:["Pantheon","Lee Sin","Vi"]},mid:{n:"Aria",kda:4.41,dpm:621,g:20,wr:55,cs:9.5,dp:23.9,gp:23.3,pool:10,ch:["Taliyah","Ahri","Twisted Fate"]},bot:{n:"Marble",kda:4.3,dpm:716,g:20,wr:55,cs:9.9,dp:27.6,gp:27.7,pool:10,ch:["Corki","Aphelios","Miss Fortune"]},sup:{n:"Vsta",kda:4.71,dpm:164,g:20,wr:55,cs:1.1,dp:6.7,gp:9.9,pool:12,ch:["Alistar","Thresh","Bard"]}}},
"G2 Esports":{elo:59.4,wr:0.73,r:{top:{n:"BrokenBlade",kda:3.29,dpm:645,g:20,wr:60,cs:8.3,dp:22.0,gp:19.5,pool:11,ch:["K'Sante","Sion","Shen"]},jng:{n:"SkewMond",kda:4.65,dpm:541,g:20,wr:60,cs:7.2,dp:19.0,gp:21.2,pool:10,ch:["Jarvan IV","Dr. Mundo","Pantheon"]},mid:{n:"Caps",kda:3.89,dpm:742,g:20,wr:60,cs:8.4,dp:25.7,gp:20.8,pool:13,ch:["Aurora","Viktor","Anivia"]},bot:{n:"Hans Sama",kda:3.43,dpm:726,g:20,wr:60,cs:9.9,dp:25.2,gp:27.6,pool:11,ch:["Varus","Corki","Yunara"]},sup:{n:"Labrov",kda:4.05,dpm:237,g:20,wr:60,cs:1.0,dp:8.1,gp:10.8,pool:13,ch:["Bard","Lulu","Nami"]}}},
"GAM Esports":{elo:19.5,wr:0.55,r:{top:{n:"Kiaya",kda:2.48,dpm:642,g:20,wr:50,cs:9.2,dp:25.3,gp:20.3,pool:11,ch:["Gnar","Ambessa","Rumble"]},jng:{n:"Draktharr",kda:2.54,dpm:383,g:20,wr:50,cs:7.0,dp:14.7,gp:18.7,pool:9,ch:["Wukong","Jarvan IV","Aatrox"]},mid:{n:"Aress",kda:3.79,dpm:537,g:20,wr:50,cs:9.6,dp:20.8,gp:21.6,pool:9,ch:["Taliyah","Ryze","Orianna"]},bot:{n:"Artemis",kda:3.79,dpm:801,g:20,wr:50,cs:10.5,dp:30.6,gp:29.4,pool:9,ch:["Yunara","Corki","Sivir"]},sup:{n:"Taki",kda:4.05,dpm:210,g:20,wr:50,cs:1.0,dp:8.6,gp:10.0,pool:12,ch:["Nautilus","Leona","Neeko"]}}},
"Galions":{elo:1.9,wr:0.53,r:{top:{n:"Carlsen",kda:5.05,dpm:657,g:20,wr:75,cs:9.0,dp:20.4,gp:19.7,pool:9,ch:["Renekton","Zaahen","K'Sante"]},jng:{n:"Thayger",kda:8.06,dpm:564,g:20,wr:75,cs:7.8,dp:17.7,gp:20.7,pool:10,ch:["Pantheon","Aatrox","Jarvan IV"]},mid:{n:"OMON",kda:5.61,dpm:812,g:20,wr:75,cs:9.3,dp:25.3,gp:21.9,pool:11,ch:["Ahri","Ryze","Aurora"]},bot:{n:"HARPOON",kda:5.56,dpm:849,g:20,wr:75,cs:10.1,dp:27.3,gp:27.2,pool:12,ch:["Ezreal","Ashe","Kai'Sa"]},sup:{n:"Zoelys",kda:6.22,dpm:284,g:20,wr:75,cs:1.0,dp:9.2,gp:10.5,pool:12,ch:["Seraphine","Bard","Neeko"]}}},
"Gen.G":{elo:77.6,wr:0.82,r:{top:{n:"Kiin",kda:4.2,dpm:791,g:20,wr:75,cs:8.9,dp:26.7,gp:20.6,pool:10,ch:["Zaahen","Renekton","Gnar"]},jng:{n:"Canyon",kda:5.97,dpm:450,g:20,wr:75,cs:7.7,dp:14.5,gp:19.7,pool:9,ch:["Vi","Pantheon","Ambessa"]},mid:{n:"Chovy",kda:7.34,dpm:820,g:20,wr:75,cs:9.9,dp:26.9,gp:22.7,pool:12,ch:["Ahri","Galio","Mel"]},bot:{n:"Ruler",kda:5.31,dpm:733,g:20,wr:75,cs:10.5,dp:24.2,gp:27.5,pool:10,ch:["Yunara","Ashe","Sivir"]},sup:{n:"Duro",kda:4.25,dpm:232,g:20,wr:75,cs:0.9,dp:7.7,gp:9.6,pool:10,ch:["Rakan","Seraphine","Neeko"]}}},
"GiantX":{elo:-11.9,wr:0.46,r:{top:{n:"Lot",kda:2.33,dpm:478,g:20,wr:40,cs:8.7,dp:20.3,gp:20.2,pool:8,ch:["K'Sante","Sion","Gnar"]},jng:{n:"ISMA",kda:3.0,dpm:385,g:20,wr:40,cs:7.0,dp:16.4,gp:19.1,pool:9,ch:["Jarvan IV","Vi","Aatrox"]},mid:{n:"Jackies",kda:2.71,dpm:636,g:20,wr:40,cs:10.1,dp:26.8,gp:22.9,pool:7,ch:["Taliyah","Orianna","Azir"]},bot:{n:"Noah",kda:3.39,dpm:665,g:20,wr:40,cs:10.3,dp:27.9,gp:27.4,pool:11,ch:["Ezreal","Aphelios","Corki"]},sup:{n:"Jun",kda:2.82,dpm:204,g:20,wr:40,cs:1.1,dp:8.6,gp:10.4,pool:11,ch:["Rakan","Bard","Neeko"]}}},
"Ground Zero Gaming":{elo:-32.5,wr:0.44,r:{top:{n:"1Jiang",kda:2.77,dpm:660,g:20,wr:55,cs:8.5,dp:23.2,gp:20.3,pool:12,ch:["Gnar","Jax","Aatrox"]},jng:{n:"Husha",kda:3.22,dpm:555,g:20,wr:55,cs:7.0,dp:19.2,gp:20.6,pool:9,ch:["Xin Zhao","Pantheon","Dr. Mundo"]},mid:{n:"JimieN",kda:3.25,dpm:653,g:20,wr:55,cs:8.7,dp:22.4,gp:20.1,pool:10,ch:["Anivia","Aurora","Orianna"]},bot:{n:"Shunn",kda:3.96,dpm:785,g:20,wr:55,cs:10.0,dp:26.7,gp:28.5,pool:10,ch:["Ashe","Yunara","Varus"]},sup:{n:"Orca",kda:3.69,dpm:251,g:20,wr:55,cs:1.1,dp:8.6,gp:10.4,pool:11,ch:["Neeko","Seraphine","Nami"]}}},
"HANJIN BRION":{elo:-34.8,wr:0.33,r:{top:{n:"Casting",kda:2.69,dpm:681,g:20,wr:35,cs:8.5,dp:24.6,gp:19.7,pool:12,ch:["Renekton","Gwen","Aurora"]},jng:{n:"GIDEON",kda:2.76,dpm:397,g:20,wr:35,cs:7.0,dp:14.0,gp:20.6,pool:10,ch:["Vi","Pantheon","Xin Zhao"]},mid:{n:"Roamer",kda:2.74,dpm:633,g:19,wr:32,cs:8.9,dp:23.1,gp:20.9,pool:10,ch:["Ryze","Taliyah","Orianna"]},bot:{n:"Teddy",kda:2.98,dpm:776,g:20,wr:35,cs:9.9,dp:27.4,gp:27.9,pool:11,ch:["Ezreal","Corki","Aphelios"]},sup:{n:"Namgung",kda:2.72,dpm:309,g:20,wr:35,cs:1.2,dp:10.9,gp:10.7,pool:13,ch:["Elise","Braum","Neeko"]}}},
"Inferno Drive Tokyo":{elo:58.1,wr:0.751,r:{top:{n:"Pinnnk",kda:3.0,dpm:638,g:20,wr:65,cs:7.7,dp:21.9,gp:18.4,pool:9,ch:["Rumble","Gnar","Illaoi"]},jng:{n:"Imagine",kda:4.89,dpm:625,g:20,wr:65,cs:7.4,dp:21.5,gp:23.2,pool:7,ch:["Qiyana","Taliyah","Nidalee"]},mid:{n:"Enapon",kda:4.26,dpm:635,g:20,wr:65,cs:8.3,dp:21.8,gp:20.0,pool:5,ch:["Ahri","Azir","Aurora"]},bot:{n:"wanan",kda:5.63,dpm:800,g:20,wr:65,cs:9.3,dp:27.2,gp:28.2,pool:7,ch:["Jinx","Ezreal","Caitlyn"]},sup:{n:"JustFocus",kda:4.44,dpm:224,g:20,wr:65,cs:0.9,dp:7.6,gp:10.2,pool:8,ch:["Nautilus","Elise","Neeko"]}}},
"Joblife":{elo:8.7,wr:0.55,r:{top:{n:"Vertigo",kda:5.76,dpm:672,g:19,wr:53,cs:9.0,dp:24.1,gp:21.5,pool:12,ch:["Rumble","Jax","Renekton"]},jng:{n:"Shift",kda:2.74,dpm:521,g:19,wr:53,cs:6.7,dp:18.6,gp:19.6,pool:9,ch:["Wukong","Pantheon","Zaahen"]},mid:{n:"Czekolad",kda:2.33,dpm:617,g:19,wr:53,cs:8.9,dp:22.3,gp:19.8,pool:6,ch:["Orianna","Taliyah","Ryze"]},bot:{n:"Comp",kda:3.93,dpm:752,g:19,wr:53,cs:10.4,dp:27.7,gp:29.0,pool:12,ch:["Varus","Kai'Sa","Corki"]},sup:{n:"HungryPanda",kda:2.34,dpm:202,g:19,wr:53,cs:1.0,dp:7.3,gp:10.1,pool:11,ch:["Karma","Alistar","Bard"]}}},
"KT Rolster":{elo:-1.7,wr:0.35,r:{top:{n:"PerfecT",kda:2.47,dpm:549,g:17,wr:35,cs:9.0,dp:21.2,gp:19.4,pool:8,ch:["Renekton","Rek'Sai","K'Sante"]},jng:{n:"Cuzz",kda:2.6,dpm:449,g:17,wr:35,cs:7.0,dp:17.7,gp:20.5,pool:11,ch:["Wukong","Vi","Dr. Mundo"]},mid:{n:"Bdd",kda:3.16,dpm:657,g:18,wr:33,cs:9.4,dp:26.1,gp:21.1,pool:8,ch:["Taliyah","Azir","Akali"]},bot:{n:"Aiming",kda:3.24,dpm:718,g:17,wr:35,cs:10.1,dp:27.5,gp:29.5,pool:10,ch:["Yunara","Lucian","Ezreal"]},sup:{n:"Ghost",kda:2.67,dpm:209,g:11,wr:36,cs:1.0,dp:7.6,gp:9.3,pool:9,ch:["Braum","Bard","Nami"]}}},
"Karmine Corp":{elo:31.1,wr:0.64,r:{top:{n:"Canna",kda:3.12,dpm:614,g:20,wr:60,cs:8.6,dp:21.9,gp:19.7,pool:12,ch:["Rumble","Sion","Gnar"]},jng:{n:"Yike",kda:3.62,dpm:474,g:20,wr:60,cs:7.2,dp:16.8,gp:19.9,pool:12,ch:["Vi","Wukong","Naafiri"]},mid:{n:"kyeahoo",kda:3.94,dpm:715,g:20,wr:60,cs:9.4,dp:25.5,gp:21.7,pool:10,ch:["Aurora","Ahri","Azir"]},bot:{n:"Caliste",kda:5.73,dpm:807,g:20,wr:60,cs:10.9,dp:28.5,gp:29.2,pool:11,ch:["Xayah","Jhin","Ezreal"]},sup:{n:"Busio",kda:4.33,dpm:205,g:20,wr:60,cs:1.0,dp:7.3,gp:9.5,pool:11,ch:["Rakan","Alistar","Karma"]}}},
"Kiwoom DRX":{elo:-7.7,wr:0.58,r:{top:{n:"Rich",kda:3.11,dpm:716,g:20,wr:50,cs:8.3,dp:24.5,gp:19.3,pool:13,ch:["K'Sante","Rumble","Kennen"]},jng:{n:"Willer",kda:3.34,dpm:484,g:20,wr:50,cs:7.0,dp:16.2,gp:19.7,pool:13,ch:["Xin Zhao","Ambessa","Jarvan IV"]},mid:{n:"Ucal",kda:3.57,dpm:633,g:20,wr:50,cs:9.4,dp:21.7,gp:22.1,pool:12,ch:["Azir","Aurora","Taliyah"]},bot:{n:"Jiwoo",kda:3.73,dpm:866,g:20,wr:50,cs:9.5,dp:29.3,gp:28.8,pool:11,ch:["Yunara","Sivir","Corki"]},sup:{n:"Andil",kda:3.17,dpm:245,g:20,wr:50,cs:1.0,dp:8.4,gp:10.2,pool:10,ch:["Nautilus","Karma","Neeko"]}}},
"L Guide Gaming":{elo:24.1,wr:0.649,r:{top:{n:"SnowRabbit",kda:1.92,dpm:678,g:20,wr:65,cs:7.7,dp:23.1,gp:19.4,pool:8,ch:["Gwen","Aatrox","Riven"]},jng:{n:"Amel",kda:3.95,dpm:521,g:20,wr:65,cs:6.6,dp:17.7,gp:20.5,pool:9,ch:["Wukong","Xin Zhao","Zaahen"]},mid:{n:"rre",kda:4.19,dpm:759,g:20,wr:65,cs:8.7,dp:25.4,gp:21.9,pool:8,ch:["Ahri","Ryze","Orianna"]},bot:{n:"NaiNa",kda:3.98,dpm:786,g:20,wr:65,cs:9.0,dp:25.8,gp:27.4,pool:10,ch:["Yunara","Ashe","Ezreal"]},sup:{n:"SaKi",kda:3.88,dpm:237,g:20,wr:65,cs:0.9,dp:8.0,gp:10.8,pool:9,ch:["Braum","Lulu","Nautilus"]}}},
"LOUD":{elo:-7.3,wr:0.7,r:{top:{n:"xyno",kda:3.85,dpm:574,g:20,wr:60,cs:9.1,dp:23.8,gp:21.9,pool:9,ch:["Ambessa","K'Sante","Sion"]},jng:{n:"YoungJae",kda:3.04,dpm:441,g:20,wr:60,cs:7.1,dp:17.7,gp:21.0,pool:10,ch:["Aatrox","Nocturne","Jarvan IV"]},mid:{n:"Envy",kda:2.71,dpm:577,g:15,wr:53,cs:8.6,dp:24.3,gp:20.8,pool:9,ch:["Aurora","Mel","Syndra"]},bot:{n:"Bull",kda:3.47,dpm:657,g:20,wr:60,cs:10.0,dp:25.9,gp:26.5,pool:13,ch:["Ezreal","Yunara","Jhin"]},sup:{n:"RedBert",kda:2.15,dpm:205,g:20,wr:60,cs:1.1,dp:8.7,gp:9.8,pool:9,ch:["Lulu","Neeko","Nami"]}}},
"LYON":{elo:11.5,wr:0.56,r:{top:{n:"Dhokla",kda:1.82,dpm:615,g:20,wr:50,cs:8.4,dp:23.1,gp:19.7,pool:13,ch:["Gnar","Aatrox","Aurora"]},jng:{n:"Inspired",kda:4.32,dpm:388,g:20,wr:50,cs:7.6,dp:14.7,gp:20.8,pool:14,ch:["Xin Zhao","Wukong","Dr. Mundo"]},mid:{n:"Saint",kda:2.57,dpm:622,g:20,wr:50,cs:9.2,dp:23.4,gp:21.6,pool:11,ch:["Azir","Ahri","Ryze"]},bot:{n:"Berserker",kda:3.77,dpm:811,g:20,wr:50,cs:10.5,dp:30.9,gp:27.8,pool:10,ch:["Ezreal","Yunara","Ashe"]},sup:{n:"Isles",kda:3.23,dpm:205,g:20,wr:50,cs:1.0,dp:7.9,gp:10.1,pool:11,ch:["Lulu","Nami","Nautilus"]}}},
"Leviatan":{elo:-18.1,wr:0.47,r:{top:{n:"Devost",kda:2.48,dpm:619,g:20,wr:50,cs:8.2,dp:23.1,gp:19.6,pool:7,ch:["Rumble","Sion","Renekton"]},jng:{n:"Booki",kda:2.79,dpm:330,g:20,wr:50,cs:6.3,dp:12.3,gp:18.1,pool:10,ch:["Vi","Jarvan IV","Xin Zhao"]},mid:{n:"Enga",kda:4.3,dpm:670,g:20,wr:50,cs:9.0,dp:26.0,gp:22.3,pool:10,ch:["Cassiopeia","Ryze","Orianna"]},bot:{n:"ceo",kda:3.48,dpm:834,g:20,wr:50,cs:9.8,dp:30.5,gp:29.3,pool:12,ch:["Kai'Sa","Sivir","Corki"]},sup:{n:"TopLop",kda:3.17,dpm:210,g:20,wr:50,cs:1.3,dp:8.1,gp:10.7,pool:11,ch:["Leona","Nautilus","Rell"]}}},
"MVK Esports":{elo:-15.0,wr:0.438,r:{top:{n:"Kratos",kda:1.8,dpm:505,g:20,wr:40,cs:8.9,dp:19.5,gp:20.3,pool:12,ch:["K'Sante","Ambessa","Aatrox"]},jng:{n:"SofM",kda:1.88,dpm:369,g:3,wr:0,cs:7.2,dp:15.2,gp:21.6,pool:3,ch:["Zac","Qiyana","Nocturne"]},mid:{n:"Kati",kda:3.22,dpm:634,g:18,wr:39,cs:9.4,dp:25.4,gp:21.9,pool:9,ch:["Orianna","Ryze","Viktor"]},bot:{n:"Shogun",kda:3.26,dpm:801,g:19,wr:42,cs:10.5,dp:31.4,gp:27.9,pool:12,ch:["Sivir","Corki","Yunara"]},sup:{n:"Elio",kda:2.44,dpm:185,g:19,wr:42,cs:1.2,dp:7.3,gp:9.7,pool:12,ch:["Alistar","Karma","Leona"]}}},
"Movistar KOI":{elo:-8.4,wr:0.55,r:{top:{n:"Myrwn",kda:2.69,dpm:575,g:20,wr:50,cs:8.6,dp:20.8,gp:19.3,pool:9,ch:["Rumble","K'Sante","Ornn"]},jng:{n:"Elyoya",kda:4.84,dpm:468,g:20,wr:50,cs:7.0,dp:16.0,gp:20.0,pool:10,ch:["Vi","Pantheon","Xin Zhao"]},mid:{n:"Jojopyun",kda:3.02,dpm:668,g:20,wr:50,cs:9.6,dp:23.9,gp:21.4,pool:10,ch:["Azir","Taliyah","Anivia"]},bot:{n:"Supa",kda:4.49,dpm:883,g:20,wr:50,cs:10.3,dp:30.9,gp:28.9,pool:8,ch:["Aphelios","Yunara","Corki"]},sup:{n:"Alvaro",kda:3.53,dpm:222,g:20,wr:50,cs:1.1,dp:8.2,gp:10.4,pool:11,ch:["Lulu","Alistar","Neeko"]}}},
"NOVEX":{elo:35.3,wr:0.657,r:{top:{n:"PonG",kda:1.69,dpm:466,g:11,wr:36,cs:7.0,dp:17.8,gp:16.6,pool:5,ch:["K'Sante","Zaahen","Sion"]},jng:{n:"Kreative",kda:4.33,dpm:550,g:20,wr:65,cs:6.8,dp:17.7,gp:22.6,pool:9,ch:["Xin Zhao","Vi","Jarvan IV"]},mid:{n:"Cryo",kda:4.56,dpm:845,g:20,wr:65,cs:8.5,dp:27.5,gp:22.3,pool:7,ch:["Ryze","Aurora","Orianna"]},bot:{n:"Aquila",kda:3.28,dpm:752,g:11,wr:36,cs:8.9,dp:28.7,gp:27.6,pool:6,ch:["Ashe","Ezreal","Aphelios"]},sup:{n:"Sh1vq",kda:4.29,dpm:255,g:20,wr:65,cs:1.0,dp:8.5,gp:10.9,pool:10,ch:["Poppy","Alistar","Rakan"]}}},
"Natus Vincere":{elo:24.9,wr:0.431,r:{top:{n:"Maynter",kda:2.89,dpm:613,g:18,wr:56,cs:8.1,dp:22.8,gp:18.5,pool:6,ch:["Sion","Rumble","K'Sante"]},jng:{n:"Rhilech",kda:4.15,dpm:507,g:18,wr:56,cs:7.4,dp:18.7,gp:21.2,pool:10,ch:["Aatrox","Ambessa","Xin Zhao"]},mid:{n:"Poby",kda:4.17,dpm:594,g:18,wr:56,cs:9.8,dp:21.8,gp:21.6,pool:8,ch:["Azir","Ryze","Taliyah"]},bot:{n:"SamD",kda:4.22,dpm:797,g:18,wr:56,cs:10.8,dp:29.3,gp:28.7,pool:9,ch:["Aphelios","Corki","Yunara"]},sup:{n:"Parus",kda:3.65,dpm:193,g:18,wr:56,cs:1.1,dp:7.4,gp:10.0,pool:10,ch:["Bard","Nautilus","Rakan"]}}},
"New Meta":{elo:-7.8,wr:0.599,r:{top:{n:"advance",kda:2.81,dpm:914,g:20,wr:55,cs:9.3,dp:30.9,gp:26.9,pool:9,ch:["Irelia","Ambessa","Gwen"]},jng:{n:"HRK",kda:2.86,dpm:466,g:20,wr:45,cs:6.8,dp:15.9,gp:20.1,pool:11,ch:["Vi","Xin Zhao","Jax"]},mid:{n:"Alps",kda:3.1,dpm:639,g:20,wr:60,cs:8.4,dp:21.3,gp:20.3,pool:7,ch:["Syndra","Orianna","Anivia"]},bot:{n:"Godot",kda:2.47,dpm:778,g:20,wr:50,cs:9.0,dp:27.3,gp:25.8,pool:12,ch:["Ezreal","Varus","Caitlyn"]},sup:{n:"Eria",kda:2.18,dpm:193,g:8,wr:38,cs:1.0,dp:7.2,gp:9.7,pool:6,ch:["Alistar","Karma","Rakan"]}}},
"Nongshim RedForce":{elo:3.9,wr:0.45,r:{top:{n:"Kingen",kda:2.5,dpm:673,g:20,wr:40,cs:8.7,dp:22.8,gp:19.1,pool:11,ch:["Renekton","Sion","Ambessa"]},jng:{n:"Sponge",kda:2.84,dpm:480,g:20,wr:40,cs:6.7,dp:16.4,gp:19.7,pool:9,ch:["Jarvan IV","Pantheon","Xin Zhao"]},mid:{n:"Scout",kda:3.55,dpm:705,g:20,wr:40,cs:9.5,dp:24.6,gp:22.5,pool:9,ch:["Taliyah","Orianna","Aurora"]},bot:{n:"Taeyoon",kda:2.45,dpm:793,g:20,wr:40,cs:9.8,dp:27.6,gp:28.5,pool:9,ch:["Yunara","Kalista","Varus"]},sup:{n:"Lehends",kda:2.86,dpm:250,g:20,wr:40,cs:1.2,dp:8.6,gp:10.1,pool:10,ch:["Renata Glasc","Leona","Alistar"]}}},
"RED Canids":{elo:-33.0,wr:0.47,r:{top:{n:"fNb",kda:2.71,dpm:605,g:20,wr:40,cs:8.8,dp:22.1,gp:21.4,pool:12,ch:["Rumble","Ambessa","Galio"]},jng:{n:"Curse",kda:3.11,dpm:436,g:20,wr:40,cs:6.2,dp:16.2,gp:18.6,pool:12,ch:["Xin Zhao","Nocturne","Vi"]},mid:{n:"Kaze",kda:4.1,dpm:703,g:20,wr:40,cs:9.0,dp:26.1,gp:23.1,pool:14,ch:["Taliyah","Tristana","Yone"]},bot:{n:"Rabelo",kda:3.6,dpm:711,g:20,wr:40,cs:9.4,dp:26.5,gp:26.2,pool:10,ch:["Jhin","Ezreal","Kai'Sa"]},sup:{n:"frosty",kda:3.4,dpm:248,g:20,wr:40,cs:1.3,dp:9.2,gp:10.8,pool:9,ch:["Rakan","Nautilus","Leona"]}}},
"Rising Gaming":{elo:36.6,wr:0.748,r:{top:{n:"YellowYoshi",kda:4.68,dpm:632,g:20,wr:70,cs:7.9,dp:18.9,gp:17.2,pool:8,ch:["Rumble","Renekton","Gnar"]},jng:{n:"ankochan",kda:3.94,dpm:454,g:10,wr:60,cs:6.4,dp:15.0,gp:19.2,pool:6,ch:["Vi","Zaahen","Ambessa"]},mid:{n:"Ramune",kda:5.23,dpm:781,g:20,wr:65,cs:8.8,dp:24.8,gp:21.6,pool:9,ch:["Ahri","Azir","Akali"]},bot:{n:"Archer",kda:5.12,dpm:1029,g:20,wr:70,cs:10.0,dp:31.6,gp:29.8,pool:9,ch:["Yunara","Kai'Sa","Ezreal"]},sup:{n:"Patch",kda:3.33,dpm:251,g:10,wr:60,cs:1.0,dp:8.6,gp:11.1,pool:7,ch:["Neeko","Bard","Rell"]}}},
"S2G Esports":{elo:-43.4,wr:0.55,r:{top:{n:"DnDn",kda:2.83,dpm:638,g:20,wr:50,cs:8.6,dp:22.3,gp:19.5,pool:10,ch:["Ambessa","Renekton","Zaahen"]},jng:{n:"Bonnie",kda:4.56,dpm:539,g:20,wr:50,cs:7.4,dp:18.3,gp:21.3,pool:9,ch:["Jarvan IV","Pantheon","Xin Zhao"]},mid:{n:"Kofte",kda:3.76,dpm:612,g:20,wr:50,cs:8.8,dp:21.7,gp:20.5,pool:12,ch:["Ryze","Galio","Ahri"]},bot:{n:"Scorth",kda:5.25,dpm:905,g:20,wr:50,cs:10.2,dp:31.3,gp:29.2,pool:11,ch:["Ezreal","Varus","Yunara"]},sup:{n:"Mersa",kda:3.32,dpm:178,g:20,wr:50,cs:1.0,dp:6.4,gp:9.4,pool:9,ch:["Karma","Nautilus","Rakan"]}}},
"SU Esports":{elo:-31.1,wr:0.38,r:{top:{n:"Dionelux",kda:2.26,dpm:557,g:20,wr:45,cs:8.0,dp:22.0,gp:18.4,pool:11,ch:["Sion","Gwen","Aatrox"]},jng:{n:"XnS",kda:4.06,dpm:548,g:20,wr:45,cs:7.7,dp:20.5,gp:23.4,pool:16,ch:["Lee Sin","Jayce","Dr. Mundo"]},mid:{n:"Fade",kda:3.2,dpm:656,g:20,wr:45,cs:9.2,dp:24.4,gp:21.8,pool:9,ch:["Cassiopeia","Azir","Yone"]},bot:{n:"Grave",kda:2.93,dpm:661,g:20,wr:45,cs:9.1,dp:24.5,gp:25.8,pool:7,ch:["Yunara","Corki","Varus"]},sup:{n:"Carry",kda:3.01,dpm:231,g:20,wr:45,cs:1.2,dp:8.6,gp:10.5,pool:10,ch:["Neeko","Leona","Rell"]}}},
"Sentinels":{elo:-81.4,wr:0.45,r:{top:{n:"Impact",kda:2.05,dpm:629,g:20,wr:35,cs:8.0,dp:24.3,gp:19.5,pool:11,ch:["Zaahen","Rumble","K'Sante"]},jng:{n:"HamBak",kda:2.88,dpm:413,g:20,wr:35,cs:6.8,dp:15.6,gp:19.8,pool:9,ch:["Wukong","Xin Zhao","Vi"]},mid:{n:"DARKWINGS",kda:3.21,dpm:745,g:20,wr:35,cs:9.2,dp:28.0,gp:22.8,pool:11,ch:["Ryze","Viktor","Taliyah"]},bot:{n:"Rahel",kda:3.43,dpm:672,g:20,wr:35,cs:9.9,dp:25.4,gp:27.3,pool:10,ch:["Yunara","Ashe","Corki"]},sup:{n:"huhi",kda:3.4,dpm:173,g:20,wr:35,cs:1.2,dp:6.8,gp:10.5,pool:10,ch:["Rell","Seraphine","Nami"]}}},
"Solary":{elo:39.8,wr:0.77,r:{top:{n:"Kryze",kda:5.67,dpm:733,g:20,wr:85,cs:9.0,dp:23.4,gp:21.1,pool:6,ch:["Gnar","Sion","Rumble"]},jng:{n:"Zicssi",kda:5.8,dpm:615,g:20,wr:85,cs:7.7,dp:19.4,gp:21.0,pool:9,ch:["Wukong","Xin Zhao","Jarvan IV"]},mid:{n:"Jool",kda:4.77,dpm:746,g:20,wr:85,cs:9.2,dp:23.6,gp:20.9,pool:11,ch:["Ahri","Taliyah","Ryze"]},bot:{n:"Aetinoth",kda:3.71,dpm:786,g:20,wr:85,cs:9.2,dp:25.1,gp:26.0,pool:10,ch:["Corki","Yunara","Ashe"]},sup:{n:"Piero",kda:4.77,dpm:266,g:20,wr:85,cs:1.0,dp:8.5,gp:10.9,pool:12,ch:["Lulu","Leona","Seraphine"]}}},
"T1":{elo:61.0,wr:0.67,r:{top:{n:"Doran",kda:2.85,dpm:764,g:20,wr:60,cs:8.8,dp:24.9,gp:20.4,pool:15,ch:["Kennen","Jayce","Gnar"]},jng:{n:"Oner",kda:4.0,dpm:553,g:20,wr:60,cs:7.0,dp:18.0,gp:20.7,pool:13,ch:["Xin Zhao","Vi","Pantheon"]},mid:{n:"Faker",kda:2.98,dpm:674,g:20,wr:60,cs:9.1,dp:22.1,gp:20.9,pool:9,ch:["Azir","Ryze","Galio"]},bot:{n:"Peyz",kda:3.79,dpm:861,g:20,wr:60,cs:10.2,dp:27.3,gp:27.7,pool:10,ch:["Aphelios","Varus","Kai'Sa"]},sup:{n:"Keria",kda:4.6,dpm:240,g:20,wr:60,cs:1.0,dp:7.8,gp:10.3,pool:13,ch:["Thresh","Rakan","Lulu"]}}},
"TLN Pirates":{elo:-2.5,wr:0.588,r:{top:{n:"Spooder",kda:2.37,dpm:576,g:20,wr:55,cs:8.0,dp:19.7,gp:19.1,pool:9,ch:["K'Sante","Sion","Ornn"]},jng:{n:"Stefan",kda:2.22,dpm:543,g:20,wr:55,cs:7.1,dp:18.9,gp:20.0,pool:12,ch:["Nocturne","Vi","Aatrox"]},mid:{n:"Toffe",kda:2.04,dpm:777,g:20,wr:55,cs:10.1,dp:27.8,gp:23.7,pool:10,ch:["Azir","Taliyah","Orianna"]},bot:{n:"Axelent",kda:3.46,dpm:772,g:20,wr:55,cs:9.7,dp:26.7,gp:27.5,pool:8,ch:["Yunara","Caitlyn","Sivir"]},sup:{n:"Thomas",kda:2.4,dpm:200,g:20,wr:55,cs:0.9,dp:7.0,gp:9.6,pool:9,ch:["Alistar","Lulu","Neeko"]}}},
"Team Heretics":{elo:-14.9,wr:0.45,r:{top:{n:"Tracyn",kda:2.57,dpm:577,g:16,wr:44,cs:9.0,dp:22.6,gp:20.9,pool:7,ch:["Ambessa","K'Sante","Rumble"]},jng:{n:"Sheo",kda:2.67,dpm:533,g:16,wr:44,cs:6.9,dp:21.0,gp:19.8,pool:10,ch:["Dr. Mundo","Xin Zhao","Pantheon"]},mid:{n:"Serin",kda:3.69,dpm:638,g:16,wr:44,cs:9.0,dp:25.8,gp:21.4,pool:8,ch:["Orianna","Azir","Aurora"]},bot:{n:"Ice",kda:3.16,dpm:610,g:16,wr:44,cs:10.4,dp:24.5,gp:28.4,pool:8,ch:["Jhin","Aphelios","Yunara"]},sup:{n:"Stend",kda:3.3,dpm:192,g:20,wr:55,cs:1.1,dp:6.9,gp:10.2,pool:12,ch:["Nautilus","Rakan","Seraphine"]}}},
"Team Secret Whales":{elo:58.5,wr:0.68,r:{top:{n:"Pun",kda:2.95,dpm:779,g:20,wr:55,cs:8.3,dp:25.6,gp:19.9,pool:12,ch:["Jax","Gwen","Ornn"]},jng:{n:"Hizto",kda:4.76,dpm:558,g:20,wr:55,cs:7.2,dp:17.9,gp:20.6,pool:9,ch:["Naafiri","Vi","Jarvan IV"]},mid:{n:"Dire",kda:3.41,dpm:701,g:20,wr:55,cs:8.5,dp:22.5,gp:20.4,pool:11,ch:["Galio","Ahri","Aurora"]},bot:{n:"Eddie",kda:4.43,dpm:838,g:20,wr:55,cs:10.2,dp:26.6,gp:28.8,pool:9,ch:["Corki","Aphelios","Yunara"]},sup:{n:"Bie",kda:4.56,dpm:226,g:20,wr:55,cs:1.2,dp:7.3,gp:10.2,pool:9,ch:["Nautilus","Rakan","Thresh"]}}},
"Team Vitality":{elo:8.9,wr:0.54,r:{top:{n:"Naak Nako",kda:2.51,dpm:719,g:18,wr:56,cs:8.6,dp:22.8,gp:21.0,pool:11,ch:["K'Sante","Rumble","Renekton"]},jng:{n:"Lyncas",kda:3.64,dpm:503,g:18,wr:56,cs:6.7,dp:15.4,gp:19.3,pool:9,ch:["Xin Zhao","Jarvan IV","Pantheon"]},mid:{n:"Humanoid",kda:2.45,dpm:776,g:18,wr:56,cs:9.5,dp:25.0,gp:22.2,pool:9,ch:["Azir","Ryze","Taliyah"]},bot:{n:"Carzzy",kda:3.29,dpm:892,g:18,wr:56,cs:9.6,dp:28.5,gp:26.8,pool:10,ch:["Corki","Ezreal","Yunara"]},sup:{n:"Fleshy",kda:3.9,dpm:251,g:18,wr:56,cs:1.1,dp:8.2,gp:10.8,pool:10,ch:["Bard","Karma","Rakan"]}}},
"Uwinks":{elo:0.6,wr:0.586,r:{top:{n:"udon",kda:2.22,dpm:472,g:19,wr:47,cs:7.6,dp:17.1,gp:17.1,pool:4,ch:["K'Sante","Renekton","Sion"]},jng:{n:"Elative",kda:2.86,dpm:717,g:19,wr:47,cs:8.3,dp:25.5,gp:26.4,pool:8,ch:["Ambessa","Gwen","Naafiri"]},mid:{n:"Jericho",kda:1.38,dpm:630,g:18,wr:11,cs:8.2,dp:23.2,gp:19.9,pool:7,ch:["Syndra","Orianna","Ryze"]},bot:{n:"Gimi",kda:2.96,dpm:910,g:19,wr:47,cs:9.4,dp:32.8,gp:29.9,pool:8,ch:["Yunara","Aphelios","Kalista"]},sup:{n:"anakin",kda:2.85,dpm:176,g:12,wr:42,cs:0.7,dp:6.8,gp:9.9,pool:4,ch:["Rakan","Lulu","Nami"]}}},
"Verdant":{elo:29.5,wr:0.82,r:{top:{n:"bobista",kda:4.36,dpm:700,g:20,wr:75,cs:8.9,dp:21.9,gp:21.5,pool:11,ch:["K'Sante","Renekton","Gangplank"]},jng:{n:"Mafro",kda:4.4,dpm:546,g:20,wr:75,cs:6.4,dp:17.5,gp:18.9,pool:9,ch:["Dr. Mundo","Nocturne","Vi"]},mid:{n:"Furuy",kda:6.58,dpm:729,g:20,wr:75,cs:8.8,dp:23.2,gp:20.9,pool:11,ch:["Orianna","Aurora","Ryze"]},bot:{n:"Mishi",kda:4.09,dpm:909,g:20,wr:75,cs:9.7,dp:28.7,gp:27.8,pool:7,ch:["Yunara","Corki","Caitlyn"]},sup:{n:"Guggu",kda:5.51,dpm:277,g:20,wr:75,cs:0.9,dp:8.8,gp:10.8,pool:9,ch:["Karma","Nautilus","Nami"]}}},
"Vivo Keyd Stars":{elo:-25.1,wr:0.37,r:{top:{n:"Boal",kda:2.67,dpm:580,g:16,wr:38,cs:8.4,dp:24.5,gp:19.4,pool:9,ch:["Rumble","Ambessa","Jax"]},jng:{n:"Disamis",kda:3.06,dpm:412,g:16,wr:38,cs:7.3,dp:17.2,gp:20.4,pool:9,ch:["Qiyana","Jarvan IV","Wukong"]},mid:{n:"Qats",kda:1.79,dpm:482,g:4,wr:25,cs:9.1,dp:23.4,gp:22.0,pool:4,ch:["Yone","Viktor","Orianna"]},bot:{n:"Morttheus",kda:3.68,dpm:645,g:16,wr:38,cs:10.3,dp:27.2,gp:28.3,pool:8,ch:["Yunara","Ezreal","Corki"]},sup:{n:"Kaiwing",kda:3.39,dpm:164,g:16,wr:38,cs:1.3,dp:7.0,gp:9.9,pool:9,ch:["Bard","Nautilus","Neeko"]}}}
};

const D = 250;
const WR_W = 1.5;
const BLUE_B = 0.15;
const LAN_B = 0.05;

const predict = (a, b, blue, lan) => {
  if (!a || !b) return 0.5;
  const diff = a.elo - b.elo;
  const base = 1 / (1 + Math.pow(10, -diff / D));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const baseC = clamp(base, 0.001, 0.999);
  let logit = Math.log(baseC / (1 - baseC));
  logit += WR_W * (a.wr - b.wr);
  if (blue) logit += BLUE_B;
  if (lan) logit += LAN_B;
  return 1 / (1 + Math.exp(-logit));
};

const POS = ["top", "jng", "mid", "bot", "sup"];
const POS_LABEL = { top: "TOP", jng: "JNG", mid: "MID", bot: "BOT", sup: "SUP" };
const POS_CLR = { top: "#ef4444", jng: "#22c55e", mid: "#3b82f6", bot: "#f59e0b", sup: "#a855f7" };

const Badge = ({ val, win }) => (
  <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${win ? "bg-green-900/60 text-green-300" : "text-gray-500"}`}>
    {typeof val === "number" ? (Number.isInteger(val) ? val : val.toFixed(1)) : val}
  </span>
);

const PlayerRow = ({ pA, pB, pos }) => {
  const cmp = (a, b) => a > b ? 1 : a < b ? -1 : 0;
  const scores = [cmp(pA.kda, pB.kda), cmp(pA.dpm, pB.dpm), cmp(pA.wr, pB.wr), cmp(pA.cs, pB.cs), cmp(pA.pool, pB.pool)];
  const aW = scores.filter(s => s > 0).length;
  const bW = scores.filter(s => s < 0).length;
  const side = aW > bW ? "a" : bW > aW ? "b" : "t";
  return (
    <div className={`grid grid-cols-[1fr_48px_1fr] gap-1 items-center py-2 px-3 rounded-lg mb-1 ${
      side === "a" ? "bg-blue-950/30 border-l-2 border-blue-500" : side === "b" ? "bg-red-950/30 border-r-2 border-red-500" : "bg-gray-800/30"
    }`}>
      <div className="text-right space-y-1">
        <div className="font-semibold text-sm text-blue-300 truncate">{pA.n}</div>
        <div className="flex justify-end gap-1.5">
          <Badge val={pA.kda} win={scores[0] > 0} />
          <Badge val={pA.dpm} win={scores[1] > 0} />
          <Badge val={pA.wr + "%"} win={scores[2] > 0} />
          <Badge val={pA.cs} win={scores[3] > 0} />
        </div>
        <div className="text-xs text-gray-600 truncate">{pA.ch.join(", ")}</div>
      </div>
      <div className="flex justify-center">
        <span className="text-xs font-bold px-2 py-0.5 rounded-full" style={{ backgroundColor: POS_CLR[pos] + "22", color: POS_CLR[pos] }}>
          {POS_LABEL[pos]}
        </span>
      </div>
      <div className="text-left space-y-1">
        <div className="font-semibold text-sm text-red-300 truncate">{pB.n}</div>
        <div className="flex gap-1.5">
          <Badge val={pB.kda} win={scores[0] < 0} />
          <Badge val={pB.dpm} win={scores[1] < 0} />
          <Badge val={pB.wr + "%"} win={scores[2] < 0} />
          <Badge val={pB.cs} win={scores[3] < 0} />
        </div>
        <div className="text-xs text-gray-600 truncate">{pB.ch.join(", ")}</div>
      </div>
    </div>
  );
};

const VERSIONS = [
  { v: "V1", auc: "0.6881", f: 63, d: "Elo, rolling stats, game context" },
  { v: "V2", auc: "0.6976", f: 79, d: "+ Player tracking, series momentum" },
  { v: "V3", auc: "0.6983", f: 106, d: "+ Coaches, travel, regional playstyle" },
  { v: "V4", auc: "0.7970", f: 130, d: "+ Patch meta, champion meta" },
];

export default function LoLPredictor() {
  const [tA, setTA] = useState("");
  const [tB, setTB] = useState("");
  const [blue, setBlue] = useState(true);
  const [lan, setLan] = useState(false);
  const [showV, setShowV] = useState(false);

  const names = useMemo(() => Object.keys(T).sort(), []);
  const a = T[tA], b = T[tB];
  const valid = a && b && tA !== tB;

  const prob = useMemo(() => valid ? predict(a, b, blue, lan) : 0.5, [a, b, blue, lan, valid]);
  const pA = Math.round(prob * 100);
  const pB = 100 - pA;

  const matchups = useMemo(() => {
    if (!valid || !a.r || !b.r) return null;
    let aW = 0, bW = 0;
    const rows = POS.map(pos => {
      const pA = a.r[pos], pB = b.r[pos];
      if (!pA || !pB) return null;
      const s = [pA.kda > pB.kda, pA.dpm > pB.dpm, pA.wr > pB.wr, pA.cs > pB.cs, pA.pool > pB.pool];
      const aw = s.filter(Boolean).length;
      if (aw >= 3) aW++; else if (aw <= 2) bW++;
      return { pos, pA, pB };
    }).filter(Boolean);
    return { rows, aW, bW };
  }, [a, b, valid]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 font-sans">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-5">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-red-400 bg-clip-text text-transparent">
            LoL Esports Predictor V4
          </h1>
          <p className="text-gray-500 text-xs mt-1">130 features | AUC 0.797 | 51,088 games (2021-2026)</p>
        </div>

        <div className="grid grid-cols-[1fr_36px_1fr] gap-2 mb-4 items-end">
          <div>
            <label className="text-xs text-blue-400 font-semibold uppercase tracking-wider">Team A</label>
            <select value={tA} onChange={e => setTA(e.target.value)}
              className="w-full mt-1 bg-gray-900 border border-blue-800/50 rounded-lg px-3 py-2 text-sm focus:border-blue-400 outline-none">
              <option value="">Select...</option>
              {names.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <span className="text-gray-600 font-bold text-center pb-2">vs</span>
          <div>
            <label className="text-xs text-red-400 font-semibold uppercase tracking-wider">Team B</label>
            <select value={tB} onChange={e => setTB(e.target.value)}
              className="w-full mt-1 bg-gray-900 border border-red-800/50 rounded-lg px-3 py-2 text-sm focus:border-red-400 outline-none">
              <option value="">Select...</option>
              {names.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
        </div>

        <div className="flex gap-3 justify-center mb-5">
          <button onClick={() => setBlue(!blue)}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${blue ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-500"}`}>
            Blue Side: A
          </button>
          <button onClick={() => setLan(!lan)}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${lan ? "bg-purple-600 text-white" : "bg-gray-800 text-gray-500"}`}>
            LAN
          </button>
        </div>

        {valid && (
          <>
            <div className="bg-gray-900 rounded-xl p-4 mb-3 border border-gray-800">
              <div className="flex justify-between mb-1.5">
                <span className={`text-xl font-bold ${pA >= 50 ? "text-blue-400" : "text-gray-500"}`}>{pA}%</span>
                <span className="text-gray-600 text-xs self-center">Win Probability</span>
                <span className={`text-xl font-bold ${pB >= 50 ? "text-red-400" : "text-gray-500"}`}>{pB}%</span>
              </div>
              <div className="h-3.5 rounded-full overflow-hidden flex bg-gray-800">
                <div className="h-full transition-all duration-500 rounded-l-full" style={{ width: `${pA}%`, background: "linear-gradient(90deg, #3b82f6, #6366f1)" }} />
                <div className="h-full transition-all duration-500 rounded-r-full" style={{ width: `${pB}%`, background: "linear-gradient(90deg, #ef4444, #dc2626)" }} />
              </div>
              <div className="flex justify-between mt-1.5 text-xs text-gray-500">
                <span>Elo {a.elo > 0 ? "+" : ""}{a.elo} | WR {(a.wr*100).toFixed(0)}%</span>
                <span>Elo {b.elo > 0 ? "+" : ""}{b.elo} | WR {(b.wr*100).toFixed(0)}%</span>
              </div>
            </div>

            {matchups && matchups.rows.length === 5 && (
              <div className="bg-gray-900 rounded-xl p-3 mb-3 border border-gray-800">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-blue-400 font-semibold text-sm truncate max-w-[40%]">{tA}</span>
                  <div className="text-center">
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Player H2H</div>
                    <div className="text-sm font-bold">
                      <span className="text-blue-400">{matchups.aW}</span>
                      <span className="text-gray-600 mx-1">-</span>
                      <span className="text-red-400">{matchups.bW}</span>
                    </div>
                  </div>
                  <span className="text-red-400 font-semibold text-sm truncate max-w-[40%] text-right">{tB}</span>
                </div>
                <div className="text-center text-xs text-gray-600 mb-1.5">KDA / DPM / WR / CS/m</div>
                {matchups.rows.map(m => <PlayerRow key={m.pos} pA={m.pA} pB={m.pB} pos={m.pos} />)}
              </div>
            )}
          </>
        )}

        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
          <button onClick={() => setShowV(!showV)} className="w-full flex justify-between items-center px-4 py-2.5 text-xs text-gray-400 hover:text-gray-200">
            <span>Version History</span>
            <span className={`transition ${showV ? "rotate-180" : ""}`}>&#9660;</span>
          </button>
          {showV && (
            <table className="w-full text-xs">
              <thead><tr className="text-gray-500 border-t border-gray-800">
                <th className="px-3 py-1.5 text-left">Ver</th><th className="px-3 py-1.5 text-right">AUC</th>
                <th className="px-3 py-1.5 text-right">#</th><th className="px-3 py-1.5 text-left">Added</th>
              </tr></thead>
              <tbody>{VERSIONS.map(v => (
                <tr key={v.v} className={`border-t border-gray-800 ${v.v === "V4" ? "bg-purple-950/30 text-purple-300" : "text-gray-400"}`}>
                  <td className="px-3 py-1.5 font-bold">{v.v}</td>
                  <td className="px-3 py-1.5 text-right font-mono">{v.auc}</td>
                  <td className="px-3 py-1.5 text-right">{v.f}</td>
                  <td className="px-3 py-1.5">{v.d}</td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
        <p className="text-center text-xs text-gray-700 mt-3">Oracle&apos;s Elixir data | Custom ML (NumPy from scratch)</p>
      </div>
    </div>
  );
}
