contraction_mapping = {"ain't": 'is not',"aren't": 'are not',"can't": 'cannot',"'cause": 'because',"could've": 'could have',"couldn't": 'could not',"didn't": 'did not',"doesn't": 'does not',"don't": 'do not',"hadn't": 'had not',"hasn't": 'has not',"haven't": 'have not',"he'd": 'he would',"he'll": 'he will',"he's": 'he is',"how'd": 'how did',"how'd'y": 'how do you',"how'll": 'how will',"how's": 'how is',"I'd": 'I would',"I'd've": 'I would have',"I'll": 'I will',"I'll've": 'I will have',"I'm": 'I am',"I've": 'I have',"i'd": 'i would',"i'd've": 'i would have',"i'll": 'i will',"i'll've": 'i will have',"i'm": 'i am',"i've": 'i have',"isn't": 'is not',"it'd": 'it would',"it'd've": 'it would have',"it'll": 'it will',"it'll've": 'it will have',"it's": 'it is',"let's": 'let us',"ma'am": 'madam',"mayn't": 'may not',"might've": 'might have',"mightn't": 'might not',"mightn't've": 'might not have',"must've": 'must have',"mustn't": 'must not',"mustn't've": 'must not have',"needn't": 'need not',"needn't've": 'need not have',"o'clock": 'of the clock',"oughtn't": 'ought not',"oughtn't've": 'ought not have',"shan't": 'shall not',"sha'n't": 'shall not',"shan't've": 'shall not have',"she'd": 'she would',"she'd've": 'she would have',"she'll": 'she will',"she'll've": 'she will have',"she's": 'she is',"should've": 'should have',"shouldn't": 'should not',"shouldn't've": 'should not have',"so've": 'so have',"so's": 'so as',"this's": 'this is',"that'd": 'that would',"that'd've": 'that would have',"that's": 'that is',"there'd": 'there would',"there'd've": 'there would have',"there's": 'there is',"here's": 'here is',"they'd": 'they would',"they'd've": 'they would have',"they'll": 'they will',"they'll've": 'they will have',"they're": 'they are',"they've": 'they have',"to've": 'to have',"wasn't": 'was not',"we'd": 'we would',"we'd've": 'we would have',"we'll": 'we will',"we'll've": 'we will have',"we're": 'we are',"we've": 'we have',"weren't": 'were not',"what'll": 'what will',"what'll've": 'what will have',"what're": 'what are',"what's": 'what is',"what've": 'what have',"when's": 'when is',"when've": 'when have',"where'd": 'where did',"where's": 'where is',"where've": 'where have',"who'll": 'who will',"who'll've": 'who will have',"who's": 'who is',"who've": 'who have',"why's": 'why is',"why've": 'why have',"will've": 'will have',"won't": 'will not',"won't've": 'will not have',"would've": 'would have',"wouldn't": 'would not',"wouldn't've": 'would not have',"y'all": 'you all',"y'all'd": 'you all would',"y'all'd've": 'you all would have',"y'all're": 'you all are',"y'all've": 'you all have',"you'd": 'you would',"you'd've": 'you would have',"you'll": 'you will',"you'll've": 'you will have',"you're": 'you are',"you've": 'you have','u.s': 'america','e.g': 'for example','colour': 'color','centre': 'center','favourite': 'favorite','travelling': 'traveling'}

punct = ['#','∅','`','¿','▲','♪','±','、','╗','à','»','・','~','，','‡','：','}','●','⊕','-','ᴵ','[','|','Â','★','—','▬','↓','†','%','▪','▼','¸','▾','←','‘','₹','▄','>','¬','❤','+',')','®','＄','{',']','<','▀','≤','â','”','♫','¢','∞','″','♥','º','¡','¥','"','é','⋅','_',';','¨','€','↑','·','^','ï','¶','─','！','╦','＆','→','“','–','₤','!','■','║','═','β','§','è','²','*','™','☆','−','―','╔','′','«','’','¾','Ã','£','¤','╚','∙','¹','=','）','▓','&','‹','½','›','│','°','π','╩','•','¦','?','÷','ə',',','√',
 "'",'（','$','╣','¼','θ','…',':','░','´','/','►','\\','％','▒','×','█','©','♦','³','Ø','¯','(','.','α','@']

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}

mispell_dict = {
    'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 
    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 
    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 
    'qoura': 'quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 
    'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'em': 'them',
    'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 
    'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'etherium': 'ethereum', 
    'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', '2k19':'2019', 'qouta': 'quota', 
    'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 
    'demonitisation': 'demonetization', 'demonitization': 'demonetization', 
    'demonetisation': 'demonetization', 'pokémon': 'pokemon', 'n*gga':'nigga', 'p*':'pussy', 
    'b***h':'bitch', 'a***h****':'asshole', 'a****le-ish':'asshole', 'b*ll-s***':'bullshit', 'd*g':'dog', 
    'st*up*id':'stupid','d***':'dick','di**':'dick',
    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", 
    "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
    "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 
    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
    "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
    "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
    "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
    "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
    'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
    'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
    '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
    'demonitization': 'demonetization', 'demonetisation': 'demonetization','\u200b': ' ', '\ufeff': '', 'करना': '', 'है': '',
    'sh*tty': 'shitty','s**t':'shit',
    'nigg*r':'nigger','bulls**t':'bullshit','n*****':'nigger',
    'p*ssy':'pussy','p***y':'pussy',
    'f***':'fuck','f*^k':'fuck','f*cked':'fucked','f*ck':'fuck','f***ing':'fucking',
    'sh*t':'shit', 'su*k':'suck', 'a**holes':'assholes','a**hole':'asshole',
    'di*k':'dick', 'd*ck': 'dick', 'd**k':'dick', 'd***':'dick',
    'bull**it':'bullshit', 'c**t':'cunt', 'cu*t':'cunt', 'c*nt':'cunt','troʊl':'trool',
    'trumpian':'bombast','realdonaldtrump':'trump','drumpf':'trump','trumpist':'trump',
    "i'ma": "i am","is'nt": "is not","‘I":'I',
    'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\
    'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\
    'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\
    'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\
    'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\
    'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\
    'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ',
    'ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop','ᴀ':'A',
    'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation',
    'doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers',
    'negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',
    'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times','garycrum':'gary crum','htmlutmterm':'html utm term',
    'RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you',
    'trumpists': 'trump', 'trumpkins': 'trump','trumpism': 'trump','trumpsters':'trump','thedonald':'trump',
    'trumpty': 'trump', 'trumpettes': 'trump','trumpland': 'trump','trumpies':'trump','trumpo':'trump',
    'drump': 'trump', 'dtrumpview': 'trump','drumph': 'trump','trumpanzee':'trump','trumpite':'trump',
    'chumpsters': 'trump', 'trumptanic': 'trump', 'itʻs': 'it is', 'donʻt': 'do not','pussyhats':'pussy hats',
    'trumpdon': 'trump', 'trumpisms': 'trump','trumperatti':'trump', 'legalizefreedom': 'legalize freedom',
    'trumpish': 'trump', 'ur': 'you are','twitler':'twitter','trumplethinskin':'trump','trumpnuts':'trump','trumpanzees':'trump',
    'justmaybe':'just maybe','trumpie':'trump','trumpistan':'trump','trumphobic':'trump','piano2':'piano','trumplandia':'trump',
    'globalresearch':'global research','trumptydumpty':'trump','frank1':'frank','trumpski':'trump','trumptards':'trump',
    'alwaysthere':'always there','clickbait':'click bait','antifas':'antifa','dtrump':'trump','trumpflakes':'trump flakes',
    'trumputin':'trump putin','fakesarge':'fake sarge','civilbot':'civil bot','tumpkin':'trump','trumpians':'trump',
    'drumpfs':'trump','dtrumpo':'trump','trumpistas':'trump','trumpity':'trump','trump nut':'trump','tumpkin':'trump',
    'russiagate':'russia gate','trumpsucker':'trump sucker','trumpbart':'trump bart', 'trumplicrat':'trump','dtrump0':'trump',
    'tfixstupid':'stupid','brexit':'<a>','Brexit':'<a>','americanophobia': '<q>', 'klastri':'<s>','thisisurl':'url','magaphants':'<x>','cheetolini':'<c>','daesh':'<b>',
    'trumpelthinskin':'<n>',
    'fuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuck':'fuck','fuckkng':'fucking','alonethank':'alone thank','answeralwaaaaaaayyyyyssssss':'answer always','aaaaaaaaaaaaaahhh':'haha','teamyoutube':'team youtube','envyness':'evilness','classssssy':'classy','thankkks':'thanks','expecteveryone':'expect everyone','hahahhahahaha':'haha','worldstarred':'world starred','moviefools':'movie fools','legaladvice':'legal advice','mildlyinteresting':'mildly interesting','quirkyyyy':'quirky','neeeeeews':'news','tooamazing':'too amazing','damitt':'damn it','buttlicker':'butt licker','everyfuckingthread':'every fucking thread','lmfaooooo':'lmfao','hungasfuck':'hungas fuck','politicalhumor':'political humor',    'questioningthe':'questioning the','thathappened':'that happened','motionwith':'motion with','fuckyoukaren':'fuck you karen','babyboys':'baby boys','yaaaaah':'ya','lmaoooooooooooooooooooo':'lmao','nononononononononono':'no','feminsts':'feminists','oldddd':'old','dammmn':'damn','rockingand':'rocking and','thisismylifenow':'this is my life now','thanksone':'thanks one','fuuuuucked':'fucked','fuckballs':'fuck balls','blackweb':'black web','playboi':'play boy','eyebleacher':'eye bleacher','justcannot':'just cannot','ohhhhthis':'oh this','fuuuuuuuuuck':'fuck','fuxked':'fucked','agaaain':'again','attentionlol':'attention lol','sciencebowl':'science bowl','knowledgebowl':'knowledge bowl','hsppening':'happening','susbscribed':'subscribed','otherrrr':'other','himmmmmmmmmmmmmm':'him','yeeeaahhh':'yeah','ahahahhahahahahahaha':'haha','younext':'you next','lmfaoooo':'lmfao','surprisedhe':'surprised he','himthank':'him thank','hahahahshshajsha':'haha','clevercomebacks':'clever comebacks','believeryeah':'believer yeah','todaylove':'today love','congratulatioins':'congratulations'
}