"""Word lists for the idea generators (ideas.py). Plain data, easy to extend: add a culture, a list, or a line."""
from __future__ import annotations

# --- names: given names by culture and gender, plus family names ------------------------------------------------------
NAMES = {
    "english": {
        "female": "Eleanor Margaret Harriet Beatrice Clara Edith Florence Lucy Agnes Mabel Winifred Dorothy Imogen Philippa Cecily Henrietta Rosalind Ada Phoebe Verity".split(),
        "male": "Edmund Thomas Arthur Walter Frederick Hugh Albert Cedric Oswald Barnaby Rupert Gideon Silas Percival Humphrey Alistair Jasper Ambrose Lionel Warren".split(),
        "family": "Ashworth Marsh Pemberton Holloway Whitlock Thorne Fairfax Blackwood Hartley Calloway Winslow Ridley Fenwick Granger Harcourt Sutton Prescott Lindsay Merrick Dunmore".split(),
    },
    "irish": {
        "female": "Siobhan Niamh Aoife Maeve Orla Saoirse Brigid Eithne Fionnuala Deirdre Roisin Caoimhe Grainne Sinead Clodagh".split(),
        "male": "Cormac Declan Eamon Fergus Liam Padraig Ronan Tadhg Brendan Colm Donal Oisin Rory Seamus Cillian".split(),
        "family": "Doyle Brennan Gallagher Keane Murphy O'Neill Quinn Rafferty Sheridan Tierney Fitzgerald Kavanagh Lynch Moran Hennessy".split(),
    },
    "scottish": {
        "female": "Morag Isla Fiona Elspeth Catriona Ailsa Moira Shona Kirsty Mairi Effie Ishbel".split(),
        "male": "Angus Duncan Hamish Ewan Fraser Lachlan Callum Alasdair Murdo Rory Fergus Struan".split(),
        "family": "MacLeod Campbell Gunn Sinclair Buchanan Drummond Grant Ross Munro Cameron Fraser Lennox Rennie Baird".split(),
    },
    "french": {
        "female": "Margaux Camille Odette Solange Genevieve Colette Elodie Amelie Delphine Isabeau Noemie Lucienne Yvette".split(),
        "male": "Laurent Étienne Mathieu Gaspard Baptiste Lucien Rémi Aurelien Thibault Anselme Fabrice Olivier Gilles".split(),
        "family": "Beaumont Duval Lefebvre Moreau Fontaine Delacroix Marchand Rousseau Lambert Garnier Villeneuve Chevalier Bellamy".split(),
    },
    "german": {
        "female": "Greta Liesel Annelise Brigitte Ingrid Hannelore Marlene Sieglinde Ottilie Katrin Friederike Lotte".split(),
        "male": "Klaus Matthias Lorenz Anselm Dieter Florian Gunther Johann Konrad Rolf Wilhelm Emil".split(),
        "family": "Brandt Keller Vogel Hartmann Lindner Albrecht Schreiber Wagner Roth Becker Engel Hoffmann Kessler".split(),
    },
    "italian": {
        "female": "Giulia Chiara Lucia Allegra Bianca Serafina Donatella Federica Elisabetta Viola Marisa Concetta".split(),
        "male": "Matteo Lorenzo Giacomo Tommaso Riccardo Dario Fabrizio Enzo Massimo Nunzio Piero Salvatore".split(),
        "family": "Moretti Bianchi Ferrara Colombo Ricci Marino Gallo Conti Bellini Caruso Barone Fontana Greco".split(),
    },
    "spanish": {
        "female": "Marisol Catalina Inés Pilar Soledad Esperanza Rocío Lucía Amparo Valentina Consuelo Alba".split(),
        "male": "Mateo Joaquín Rafael Santiago Alejandro Emilio Íñigo Tomás Rodrigo Gabriel Ignacio Álvaro".split(),
        "family": "Navarro Castillo Herrera Salazar Mendoza Ortega Vargas Fuentes Rivera Delgado Cabrera Montoya Quintero".split(),
    },
    "russian": {
        "female": "Anastasia Svetlana Irina Tatiana Galina Yelena Larisa Oksana Zinaida Nadezhda Vera Lyudmila".split(),
        "male": "Dmitri Alexei Nikolai Pavel Viktor Sergei Boris Anatoly Grigori Mikhail Yuri Konstantin".split(),
        "family": "Volkov Petrov Sokolov Morozov Kuznetsov Orlov Zaitsev Popov Smirnov Lebedev Romanov Antonov Kovalenko".split(),
    },
    "polish": {
        "female": "Agnieszka Halina Krystyna Wanda Zofia Jadwiga Marzena Danuta Ewa Barbara Irena".split(),
        "male": "Tadeusz Kazimierz Stanisław Wojciech Jerzy Zbigniew Bogdan Marek Janusz Leszek Henryk".split(),
        "family": "Kowalski Nowak Wójcik Zielinski Kaminski Lewandowski Szymanski Dabrowski Jankowski Mazur Kozlowski".split(),
    },
    "greek": {
        "female": "Eleni Katerina Sophia Despina Ariadne Penelope Calliope Thalia Irini Melina Daphne Xenia".split(),
        "male": "Nikos Dimitris Stavros Yannis Theodoros Alexandros Kostas Leonidas Orestis Spyros Vassilis".split(),
        "family": "Papadakis Georgiou Karras Stavrou Alexiou Mavros Nikolaidis Kouros Vlachos Laskaris Pappas".split(),
    },
    "japanese": {
        "female": "Haruka Sakura Yuki Akiko Midori Hana Kaori Noriko Setsuko Tomoe Mizuki Ayame".split(),
        "male": "Haruto Kenji Takeshi Daichi Hiroshi Ryo Satoru Isamu Kaito Shinji Tadashi Masaru".split(),
        "family": "Tanaka Sato Kobayashi Nakamura Yamada Matsuda Fujimoto Okada Hayashi Mori Shimizu Inoue Kuroda".split(),
    },
    "chinese": {
        "female": "Mei Lian Xiu Yun Ling Hua Jing Fang Shan Qing Lan Ying".split(),
        "male": "Wei Jun Hao Ming Tao Bo Cheng Lei Feng Yong Kai Zhi".split(),
        "family": "Wang Li Zhang Liu Chen Yang Huang Zhao Wu Zhou Xu Sun Ma Zhu".split(),
    },
    "korean": {
        "female": "Min-ji Seo-yeon Ji-woo Eun-ha Soo-jin Hye-won Da-eun Yuna Bo-ra Chae-won".split(),
        "male": "Min-jun Ji-ho Joon-seo Hyun-woo Sung-min Tae-yang Do-yun Seung-ho Jae-hyun Woo-jin".split(),
        "family": "Kim Lee Park Choi Jung Kang Cho Yoon Jang Lim Han Shin".split(),
    },
    "indian": {
        "female": "Anjali Priya Meera Kavita Lakshmi Sunita Radha Ishani Devika Nandini Tara Padma".split(),
        "male": "Arjun Rohan Vikram Anand Suresh Rajesh Kiran Dev Ishaan Mohan Prakash Aditya".split(),
        "family": "Sharma Iyer Banerjee Kapoor Reddy Menon Chatterjee Verma Pillai Desai Joshi Rao Mehta".split(),
    },
    "arabic": {
        "female": "Layla Amira Yasmin Salma Noor Rania Dalia Samira Farah Zainab Huda Mariam".split(),
        "male": "Omar Karim Yusuf Tariq Faisal Hassan Idris Samir Khalid Nabil Rashid Jamal".split(),
        "family": "Haddad Mansour Nasser Khalil Rahman Saleh Aziz Farouk Darwish Habib Bakr Sabbagh".split(),
    },
    "west_african": {
        "female": "Adaeze Folasade Amara Ngozi Yetunde Chioma Abena Ama Efua Titilayo Nneka Zainab".split(),
        "male": "Chidi Olumide Kwame Kofi Emeka Tunde Femi Kwabena Obinna Seun Yaw Babatunde".split(),
        "family": "Okonkwo Adeyemi Mensah Boateng Nwosu Balogun Owusu Eze Adebayo Asante Okafor Danso".split(),
    },
    "swedish": {
        "female": "Karin Linnea Astrid Ingeborg Greta Maja Sigrid Elsa Britta Signe Hilda Freja".split(),
        "male": "Anders Nils Gustaf Erik Lars Olof Henrik Axel Sven Per Carl Johan".split(),
        "family": "Lindqvist Sjoberg Andersson Johansson Nilsson Larsson Eriksson Lundgren Bergstrom Holmberg Ekstrom Wallin".split(),
    },
    "norse": {
        "female": "Astrid Sigrid Freya Ingrid Solveig Thora Gudrun Ragnhild Helga Ylva Signe Eira".split(),
        "male": "Bjorn Leif Ragnar Sven Eirik Torvald Halvard Gunnar Olaf Ivar Hakon Ulf".split(),
        "family": "Haraldsen Ostergaard Lindqvist Bergstrom Thorsen Nilsson Eriksen Solberg Falk Rask Dahl".split(),
    },
    "welsh": {
        "female": "Cerys Bronwen Megan Rhiannon Gwen Eirlys Carys Angharad Seren Nia".split(),
        "male": "Rhys Gareth Dafydd Emrys Owain Iolo Gethin Idris Tomos Evan".split(),
        "family": "Pryce Llewellyn Powell Gruffydd Bevan Vaughan Rees Pritchard Morgan Gethin".split(),
    },
}
# gender "neutral" draws from either list; these are extra neutral-leaning given names
NEUTRAL_GIVEN = "Robin Sam Alex Jordan Riley Morgan Quinn Avery Rowan Sage Ash Wren Kit Lane Jesse Dale Tate Reese".split()

# syllable generators for invented names
FANTASY = {
    "elvish": (["Ael", "Ily", "Fae", "Lor", "Sil", "Ele", "Mir", "Thal", "Nae", "Cal", "Ari", "Lue"], ["rien", "wen", "thas", "ndil", "lanor", "vael", "iel", "ssa", "nys", "dor", "lith", "phine"]),
    "dwarvish": (["Bor", "Dur", "Thrain", "Gim", "Kaz", "Brom", "Dag", "Orm", "Hald", "Grum", "Bal", "Thor"], ["in", "gar", "rik", "bur", "dun", "mund", "li", "nor", "grim", "dak", "din", "kar"]),
    "harsh": (["Skar", "Vor", "Kra", "Mog", "Zul", "Drak", "Gor", "Rask", "Thok", "Urg", "Naz", "Brul"], ["gash", "tok", "mar", "dul", "zek", "nak", "grul", "vek", "thar", "gor", "sk", "ush"]),
    "soft": (["Mae", "Lio", "Sa", "Ni", "Ela", "Tia", "Oli", "Ysa", "Fio", "Lena", "Ami", "Isa"], ["ra", "lune", "belle", "nia", "mere", "sia", "line", "wyn", "lia", "ra", "la", "ven"]),
}

# --- places ------------------------------------------------------------------------------------------------------------
PLACE_PARTS = {
    "village": (["Ash", "Brook", "Cold", "Dun", "Elm", "Fair", "Grey", "Hollow", "Kings", "Little", "Mill", "North", "Old", "Pine", "Red", "Stone", "Thorn", "West", "Willow", "Wren"],
                ["ford", "ton", "ham", "wick", "bury", "combe", "field", "worth", "stead", "by", "thorpe", "dale", "mere", "borough", "ley", "cross"]),
    "town": (["Port", "Saint", "New", "Upper", "Lower", "East", "Castle", "Market", "Bridge", "Harbour", "Iron", "Salt"],
             ["haven", "bridge", "gate", "minster", "mouth", "chester", "marsh", "heath", "wharf", "green", "well", "shore"]),
}
PLACE_LISTS = {
    "inn": ("The {a} {b}", ["Crooked", "Gilded", "Drowned", "Laughing", "Sleeping", "Rusty", "Silver", "Blind", "Merry", "Hanged", "Nodding", "Weary"], ["Heron", "Lantern", "Anchor", "Fiddle", "Badger", "Crown", "Oar", "Magpie", "Kettle", "Mermaid", "Pilgrim", "Candle"]),
    "ship": ("The {a} {b}", ["Brave", "Pale", "Restless", "Salt", "Quiet", "Golden", "Last", "Second", "Wandering", "Faithful", "Hungry", "Lucky"], ["Margaret", "Sparrow", "Tern", "Promise", "Harvest", "Gull", "Meridian", "Comfort", "Resolve", "Marigold", "Albatross", "Dawn"]),
    "street": ("{a} {b}", ["Candle", "Tanners", "Cobblers", "Ropewalk", "Chapel", "Weavers", "Fish", "Pudding", "Bell", "Lamplighters", "Coopers", "Gaol"], ["Lane", "Row", "Street", "Alley", "Close", "Yard", "Passage", "Hill", "Walk", "Court"]),
    "wild": ("The {a} {b}", ["Whispering", "Broken", "Drowned", "Hollow", "Silent", "Sunken", "Burning", "Forgotten", "Glass", "Black", "Thousand", "Weeping"], ["Wood", "Marsh", "Fells", "Moor", "Reach", "Downs", "Fen", "Gorge", "Strand", "Tors", "Weald", "Barrens"]),
}

# --- character ingredients ------------------------------------------------------------------------------------------------------
OCCUPATIONS = ("night-train conductor, lighthouse keeper, apothecary, cartographer, piano tuner, ferryman, bell-ringer, chimney sweep, clockmaker, "
               "beekeeper, midwife, customs officer, bookbinder, tailor, stonemason, school teacher, telegraph operator, glassblower, shepherd, "
               "ship's cook, undertaker, fishmonger, translator, locksmith, watchmaker, gravedigger, innkeeper, bailiff, court reporter, "
               "taxidermist, tram driver, archivist, florist, boat builder, lamplighter, tax collector, cheesemonger, hotel porter, seamstress, "
               "retired boxer, travelling dentist, radio announcer, museum guard, flower-seller, ice-cream seller, steeplejack").split(", ")
QUIRKS = ("hums when nervous; keeps a notebook of every stranger they meet; never sits with their back to a door; collects buttons; "
          "talks to plants; counts stairs; always arrives early and says nothing about it; saves the best of every meal for last; "
          "tells small lies about their age; hoards string; cannot whistle and tries constantly; writes letters they never send; "
          "knows the name of every dog on the street; refuses to say goodbye; laughs a second too late; tests every chair before sitting; "
          "memorises train timetables; folds everything into neat squares; names the objects they own; speaks to the dead as if they were in the next room; "
          "answers questions with another question; keeps one glove and loses the other on purpose").split("; ")
FLAWS = ("proud; jealous of small things; cannot admit being wrong; avoids every difficult conversation; says yes and means no; "
         "holds grudges for decades; always the martyr; needs to be liked by everyone; too sure of their own cleverness; "
         "judges people by their shoes; collects other people's secrets and uses them; cannot stand silence; reckless with money; "
         "afraid of being ordinary; mistakes kindness for weakness; tells the truth at the worst moment; lets others take the blame; "
         "quick to anger and quicker to forget; hides behind jokes; clings to the way things used to be").split("; ")
SECRETS = ("has never learned to read; is not who the town thinks they are; owes a dangerous favour; saw something they should not have; "
           "has been sending money to someone they swore to forget; once left someone behind; forged a signature long ago; "
           "is far older than they look; knows where the missing person is; was once famous under another name; "
           "has a second family in the next county; kept something that was not theirs; broke a promise to the dying; "
           "is quietly ill and telling no one; wrote the letter everyone is looking for; started the fire").split("; ")
WANTS = ("to be forgiven; to be left alone; to see the sea once more; to prove someone wrong; to finish what a parent started; to be chosen; "
         "to get home; to find a particular person; to keep the family business alive; to be believed; to learn the truth about a death; "
         "to earn enough to leave; to be someone's first thought; to win one honest victory; to stop running; to be remembered").split("; ")
FEARS = ("being forgotten; deep water; being found out; the dark at the top of stairs; growing old alone; fire; open crowds; "
         "being poor again; a particular name being spoken; silence in a house; losing their memory; being useless; "
         "a locked door; birds in the house; being touched without warning; going home").split("; ")
HABITS = ("drinks tea so strong it stains the cup; walks the same route at the same hour; taps three times before entering a room; "
          "reads the last page first; sleeps in a chair, never a bed; polishes shoes late at night; writes lists and loses them; "
          "feeds a stray no one else will touch; sings under their breath; wipes the table twice; rises before the birds; "
          "says grace over every meal, even bread").split("; ")

# --- plot ---------------------------------------------------------------------------------------------------------------------------
PROBLEMS = ("a letter arrives that should not exist; the one witness changes their story; a bridge washes out and no one can leave; "
            "someone returns who was thought dead; a debt comes due on the worst possible day; two friends want the same thing; "
            "a rule everyone follows turns out to have no origin; the map is wrong and someone knew; a child says something they could not know; "
            "an important object is found where it cannot be; the one honest person is caught lying; a storm closes the only road; "
            "a stranger knows everyone's name; the house has a room nobody remembers; a promise made years ago is called in; "
            "the help that was promised does not come; a wedding, a funeral and a trial fall on the same day; "
            "someone has been wearing a dead man's coat; the money is gone and so is the person who counted it; "
            "a rival offers exactly what is needed, at a price; an old photograph shows a face that has not aged").split("; ")
TWISTS = ("the narrator has been wrong about who they were talking to; the helper was the cause; the missing person left on purpose; "
          "the villain is protecting someone; the gift was a warning; the two events are one event seen from opposite sides; "
          "the quiet one has been in charge all along; the thing everyone feared is smaller than the thing they ignored; "
          "the letter was never sent by the person who signed it; the cure and the poison are the same; "
          "what looked like an ending is the first day of something else; the witness was the culprit's alibi and the culprit's victim").split("; ")
SENSES = ("the smell of wet wool and coal smoke; a clock that is always four minutes slow; the taste of tin at the back of the throat; "
          "a floorboard that answers every footstep; sunlight through a dirty window, thick as syrup; the hiss of a kettle in another room; "
          "frost patterns spreading on the inside of the glass; the sweet rot of apples in a cellar; someone humming off-key behind a wall; "
          "the cold of a doorknob in winter; a bell heard across water; the rasp of a match; rain on a tin roof; "
          "bread baking two streets away; the grit of sand in every fold of clothing; a dog barking, far off, then stopping").split("; ")
WEATHER = ("thin rain that never quite becomes a storm; a white fog that swallows the end of the street; bright, hard cold; "
           "heavy heat before thunder; wind that rattles every loose thing; first snow, still falling; a clear night with too many stars; "
           "drizzle on the glass all morning; a pale, flat sky; the damp that gets into the books; a sudden warm day in the wrong month").split("; ")
OBJECTS = ("a brass key with no lock; a ledger with pages torn out; a child's shoe on a windowsill; a half-finished letter; a cracked pocket watch; "
           "a map with one town crossed out; a jar of buttons; a tin of foreign coins; an unsigned painting; a pair of spectacles that are not theirs; "
           "a locked music box; a coat with something sewn into the lining; a ticket for a train that no longer runs; "
           "a photograph with a face cut out; a bell with no clapper").split("; ")
CONSTRAINTS = ("Write the next scene from the point of view of the person least involved. | Let two characters talk about something other than what they mean. | "
               "Begin in the middle of an argument. | Use no adjectives for one whole paragraph. | Let a character be right for the wrong reason. | "
               "Slow the scene down: one minute takes a page. | Cut the scene where the reader expects it, then start after the consequences. | "
               "Give the quietest character the best line. | Describe the room only through what is missing from it. | "
               "Let the weather be a character who wants something. | Make one detail slightly wrong and let a character notice. | "
               "Tell it as a rumour passed along by three different people. | End the paragraph on an object, not a feeling. | "
               "Let someone do a kindness for a selfish reason. | Start with the smell of the place. | "
               "Have a character change the subject, twice, to avoid something.").split(" | ")
TITLE_PATTERNS = ("The {a} of {b}; The {b} {a}; {b} at {c}; A {a} for {b}; The Last {b}; Where the {b} {v}; {b} and the {a}; The {a} Hour; "
                  "What the {b} Knew; Beyond the {b}; The {c} {b}").split("; ")
TITLE_WORDS = {
    "a": "Silence Weight Keeper Garden Lantern Winter Mercy Ledger Harvest Reckoning Hollow Thread Orchard Promise Crossing".split(),
    "b": "Lighthouse Stranger Tide Clockmaker Letters Orchard Station Heron River Widow Bell Window Frontier Garden Harbour".split(),
    "c": "Midnight Hollow Salt Ember Glass Marrow Winter Paper Lantern Iron Cinder Thistle".split(),
    "v": "Rivers Bells Lamps Birds Roads Stars".split(),
}
SPARK_KINDS = ["name", "object", "sense", "problem", "constraint", "weather", "quirk"]
