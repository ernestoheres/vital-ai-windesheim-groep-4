# %% [markdown]
# # Analyse van het probleem
# ## Vraagstelling
# Hoe kan Isala ziekenhuis met behulp van een model sepsis vroegtijdig voorspellen, zodat zorgverleners sneller kunnen ingrijpen, de zorgkwaliteit verbetert en de werkdruk op verpleegkundigen en artsen afneemt?
# 
# ## Het probleem
# Sepsis is een levensbedreigende aandoening die ontstaat wanneer het lichaam extreem reageert op een infectie. Deze ontregelde afweerreactie kan leiden tot orgaanfalen, shock en overlijden. In Nederland ontwikkelen jaarlijks tienduizenden patiënten sepsis. Een snelle herkenning is essentieel: ieder uur vertraging in behandeling verhoogt de mortaliteit met ongeveer 4–8%.
# 
# In de klinische praktijk is vroege herkenning echter lastig. Symptomen zoals verhoogde hartslag, koorts, lage bloeddruk of verwardheid zijn vaak niet specifiek en kunnen ook bij andere aandoeningen voorkomen. Hierdoor wordt sepsis regelmatig pas laat herkend, terwijl juist de eerste uren cruciaal zijn voor behandeling en herstel.
# 
# Tegelijkertijd staat de zorgsector onder toenemende druk door personeelstekorten, stijgende zorgvraag en een groeiende administratieve belasting. Zorgverleners moeten continu grote hoeveelheden patiëntdata interpreteren, wat het risico op menselijke fouten of vertraagde signalering vergroot.
# 
# **Binnen de strategische visie van Isala ziekenhuis speelt digitale innovatie een belangrijke rol. Het ziekenhuis wil technologie inzetten om:**
# - Patiënten meer regie te geven
# - Digitale gastvrijheid te verbeteren
# - Veiligere zorg te realiseren
# - Zorguitkomsten te verbeteren
# - Processen slimmer te digitaliseren
# - De werkdruk van zorgprofessionals te verlagen
# 
# Een voorspellend model voor sepsis sluit direct aan op deze doelstellingen.
# 
# ## Doel van het onderzoek
# Het doel van dit onderzoek is het ontwikkelen van een betrouwbaar model dat sepsis in een vroeg stadium kan voorspellen op basis van patiëntgegevens.
# 
# **Het model moet in staat zijn om:**
# - Sepsis minimaal zes uur vóór de klinische classificatie te signaleren
# - Realtime risicovoorspellingen te genereren
# - Zorgverleners tijdig te ondersteunen bij klinische besluitvorming
# - Het aantal gemiste sepsisgevallen te verminderen
# - Tegelijkertijd het aantal foutieve alarmen (false positives) beperkt te houden
# 
# ## Hypothese
# Wanneer een machine learning model wordt getraind op historische patiëntgegevens, vitale waarden en laboratoriumuitslagen, kan het patronen herkennen die voorafgaan aan sepsis.
# 
# **De verwachting is dat:**
# - Het model sepsis minimaal zes uur eerder kan voorspellen dan de huidige klinische detectie
# - Vroege signalering leidt tot snellere behandeling
# - Dit resulteert in betere patiëntuitkomsten, lagere mortaliteit en minder IC-opnames
# - Het model zorgprofessionals ondersteunt zonder een onacceptabele toename van alarmmoeheid te veroorzaken
# 
# **Het succes van het model hangt af van de balans tussen:**
# - **Sensitiviteit** (zoveel mogelijk echte sepsisgevallen herkennen)
# - **Specificiteit** (onnodige waarschuwingen beperken)
# 
# # Ethiek
# Het gebruik van AI binnen de zorg brengt belangrijke ethische vraagstukken met zich mee. Hoewel voorspellende modellen grote voordelen kunnen bieden, mogen medische beslissingen nooit volledig afhankelijk worden van algoritmes.
# 
# Belangrijke aandachtspunten zijn:
# 
# **Privacy en databescherming**
# Patiëntgegevens bevatten gevoelige medische informatie. Daarom moet het onderzoek voldoen aan:
# - De AVG/GDPR-richtlijnen
# - Veilige dataopslag
# - Anonimisering van patiëntdata
# - Strikte toegangscontrole
# 
# **Transparantie en uitlegbaarheid**
# Zorgverleners moeten begrijpen waarom het model een patiënt als risicovol markeert. Daarom zijn uitlegbare modellen belangrijk, bijvoorbeeld door inzicht te geven in welke vitale waarden of labresultaten bijdragen aan een voorspelling.
# 
# **Bias en eerlijkheid**
# Het model mag geen systematische verschillen veroorzaken tussen patiëntgroepen op basis van leeftijd, geslacht, afkomst of medische achtergrond. Daarom moet het model getest worden op bias en generaliseerbaarheid.
# 
# **Verantwoord gebruik van model**
# Het model dient als beslisondersteuning en niet als vervanging van klinisch oordeel. De eindverantwoordelijkheid blijft altijd bij de arts of verpleegkundige.
# 
# **Alarmmoeheid**
# Te veel foutieve meldingen kunnen leiden tot verminderde alertheid bij zorgverleners. Daarom is het essentieel om een goede balans te vinden tussen gevoeligheid en betrouwbaarheid.
# 
# # Cycle I
# In deze cyclus ligt de focus eerst op een aantal basismodellen, gecombineerd met een zo zorgvuldig mogelijk uitgevoerde data-preparatie. Het nadeel hiervan is echter dat pas achteraf blijkt hoe effectief deze aanpak daadwerkelijk is, namelijk nadat het model is ontwikkeld en gevalideerd aan de hand van de utility score.
# 
# ## Voorbereiding
# Voor deze stap heb ik alvast een aantal voorbereidingen getroffen om het werken met de dataset te vereenvoudigen. Het bepalen van sepsis gebeurt op basis van twee scores: de qSOFA-score en de SOFA-score. In de dataset (`test_data.csv`) ontbreken echter enkele variabelen die nodig zijn om deze scores volledig volgens de standaarddefinities te berekenen. Hieronder volgt een overzicht van de beperkingen:
# 
# **qSOFA-score**
# De qSOFA-score heeft normaal een bereik van 0–3, maar in deze dataset is dit beperkt tot 0–2, omdat:
# - De mentale status (Glasgow Coma Scale, GCS) ontbreekt.
# 
# **SOFA-score**
# Ook voor de SOFA-score zijn niet alle componenten volledig beschikbaar:
# - Het centrale zenuwstelsel (CNS) kan niet worden meegenomen, aangezien de GCS ontbreekt.
# - Voor respiratie wordt de verhouding SpO₂/FiO₂ gebruikt in plaats van de gebruikelijke PaO₂/FiO₂. Deze benadering is minder nauwkeurig.
# - De cardiovasculaire (CV) score is slechts gedeeltelijk beschikbaar. Alleen de Mean Arterial Pressure (MAP) wordt gebruikt, waardoor deze component maximaal 1 punt kan bijdragen, terwijl dit normaal kan oplopen tot 4 punten op basis van meerdere metingen.
# 
# Om consistent met deze beperkingen om te gaan, is de klasse `SofaCalculator` ontwikkeld. Deze klasse bevat alle logica voor het berekenen van de qSOFA- en SOFA-scores op basis van de beschikbare data. Door deze klasse te importeren in de notebooks, kunnen alle groepsleden dezelfde berekeningsmethode hanteren en wordt inconsistentie in de analyses voorkomen.

# %% [markdown]
# De gebruikte modules bevinden zich op een andere locatie binnen de root van de folderstructuur dan de notebooks. Daarom moet deze locatie expliciet worden toegevoegd aan de zoekpaden die Python gebruikt om imports te vinden.

# %%
import sys
import os

sys.path.append(os.path.abspath("../../src"))

# %% [markdown]
# Hieronder worden alle imports opgenomen die nodig worden geacht voor de eerste cyclus.

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from helpers.notebook_helpers import read_dataset

df = read_dataset()

# %% [markdown]
# De dataset bevat een groot aantal variabelen, namelijk 43 kolommen. Het merendeel hiervan bestaat uit medische metingen die bij patiënten zijn uitgevoerd.
# 
# Daarnaast zijn er twee kolommen die betrekking hebben op tijd:
# - **HospAdmTime:** Meet de tijd tussen de opname in het ziekenhuis en de opname op de Intensive Care
# - **hour:** Het uur van de desbetreffende ziekenhuis bezoek per rij in de dataset 
# 
# Verder bevat de dataset één kolom zonder naam, waardoor onduidelijk is welke variabele deze vertegenwoordigt en hoe de bijbehorende waarden geïnterpreteerd moeten worden. Ook de kolom `Gender` is niet eenduidig te interpreteren, aangezien de codering van de waarden niet is gespecificeerd.
# 
# Op basis van een eerste verkenning van de data lijken de meeste waarden binnen een realistisch bereik te vallen. Hoewel ik geen medisch specialist ben, lijken eventuele uitschieters mogelijk binnen een ziekenhuiscontext, waar extreme waarden in sommige gevallen kunnen voorkomen.
# 
# Wat wel opvalt, is dat de `hour`-kolom waarden bevat tot maximaal 335. Het is niet direct duidelijk hoe deze geïnterpreteerd moeten worden, waardoor het lastig is om te beoordelen of deze waarden correct zijn. Dit brengt enige onzekerheid met zich mee over de interpretatie en betrouwbaarheid van deze specifieke tijdsvariabele.
# 
# Ook valt het op dat er veel NaN waardes zijn. Dit kan nutuurlijk zijn omdat de waardes niet gemeten zijn tijdens het ziekenhuis bezoek.

# %%
df.info()
df.describe()

# %%
df.head()

# %% [markdown]
# Uit deze kolom blijkt dat er zowel boven als onder de maximale waarde veel uitschieters (outliers) aanwezig zijn. Bij `Fibrinogen` is geen vergelijkbaar patroon zichtbaar, al valt dit in de visualisatie deels buiten het diagram om de leesbaarheid te behouden.
# 
# Zoals eerder benoemd, is het goed mogelijk dat deze outliers daadwerkelijk valide waarden zijn, aangezien het om medische metingen in een ziekenhuisomgeving gaat, waar extreme waarden kunnen voorkomen. Tegelijkertijd roept de hoeveelheid outliers ook twijfel op over de betrouwbaarheid of interpretatie van sommige metingen.

# %%
plt.figure(figsize=(14, 12))

sns.boxplot(data=df.drop(columns=['Patient_ID', 'Hour', 'HospAdmTime', 'Unnamed: 0', 'Unit1', 'Unit2']))
plt.ylim(-20, 450)
plt.xticks(rotation=90)

# %% [markdown]
# In de kolom `Gender` komen uitsluitend de waarden [0, 1] voor, wat erop wijst dat de variabele gecodeerd is. Hierdoor is op dit moment niet direct te achterhalen welk getal bij welk geslacht hoort binnen deze dataset.
# 
# Wel is er een lichte scheve verdeling zichtbaar: waarde 1 komt voor in 54,89% van de gevallen, terwijl waarde 0 in 45,11% van de dataset voorkomt.

# %%
df['Gender'].unique()

# %%
sns.countplot(x=df['Gender'])

# %% [markdown]
# In de eerdere visualisatie was al zichtbaar dat waarde 1 in de `Gender`-kolom vaker voorkomt dan waarde 0. Uit nieuwsgierigheid is vervolgens precies berekend hoe groot dit verschil is in procenten.

# %%
genderLenght = len(df['Gender'])
gender_vals = df['Gender'].unique()

def calc_percentage(val: int) -> float:
    return (val /genderLenght) * 100

for val in gender_vals:
    count = (df['Gender'] == val).sum()
    print(f'{val}: {calc_percentage(count):.2f}%')

# %% [markdown]
# Uit dit pariplot durf ik geen conclusies te trekken omdat deze gigantisch is. Een heatmap lijkt mij veel geschikter om de correlatie tussen verschillende gevens te bekijken.
# 
# **Deze is op het moment van inleveren uitgeschakeld (gecomment), omdat de Python-kernel anders crasht en de rest van het notebook daardoor niet meer kan worden uitgevoerd.**

# %%
# sns.pairplot(data=df.drop(columns=['Patient_ID', 'Hour', 'HospAdmTime', 'Unnamed: 0', 'Unit1', 'Unit2']))   

# %% [markdown]
# De heatmap hieronder is overzichtelijker dan de pairplot die daarna volgt. Er is gekozen voor een driehoekige weergave, zodat dubbele waarden niet worden getoond en de visualisatie compacter blijft.
# 
# Annotaties zijn weggelaten, omdat deze de grafiek opnieuw onoverzichtelijk zouden maken, vergelijkbaar met de pairplot.

# %%
corr = data=df.drop(columns=['Patient_ID', 'Hour', 'HospAdmTime', 'Unnamed: 0', 'Unit1', 'Unit2']).corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14, 12))

sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm')

# %% [markdown]
# Zoals te zien is, bevat de dataset meerdere observaties per Patient_ID, wat erop wijst dat per patiënt meerdere metingen over de tijd zijn vastgelegd.

# %%
df.duplicated(subset=['Patient_ID']).sum()

# %%
y_test_data = df[['Patient_ID', 'HospAdmTime', 'Hour']].sort_values(by=['Patient_ID', 'Hour'])
y_test_data.head(12)

# %%
(y_test_data['Patient_ID'] == 2).sum()

# %% [markdown]
# In deze cyclus wordt afgezien van het gebruik van tijdsreeksen en ligt de nadruk op het toepassen van modellen zoals decision trees en random forests. In een volgende cyclus zal de data verder worden verrijkt. Hierbij zal per meting en per `Patient_ID` een tijdsverschil worden berekend, zodat modellen ontwikkeld kunnen worden die het tijdsverloop expliciet meenemen.

# %% [markdown]
# ## Data preperation
# ### Clean Data
# De kolom `Unnamed: 0` heeft geen toegevoegde waarde en wordt daarom verwijderd. Het is bekend dat deze kolom een uurwaarde representeert, maar deze informatie is al aanwezig in een aparte kolom. Om redundantie te voorkomen, wordt `Unnamed: 0` niet meegenomen in de verdere analyse.
# 
# Daarnaast worden ontbrekende waarden (null values) vervangen door 0. Dit is noodzakelijk omdat de gebruikte modules voor het visualiseren van de decision tree en de random forest geen null-waarden ondersteunen. Aangenomen wordt dat deze ontbrekende waarden het gevolg zijn van metingen die tijdens het ziekenhuisbezoek niet zijn uitgevoerd. Door deze waarden op 0 te zetten, wordt impliciet vastgelegd dat de meting niet heeft plaatsgevonden.

# %%
df = df.drop(columns=['Unnamed: 0'])
df = df.fillna(0)

# %% [markdown]
# ### Construct Data
# De enige aanvullende kolommen die nodig zijn, worden berekend en toegevoegd aan de dataset door de `SofaCalculator`.

# %%
from scepsis_prediction.sofa_calculator import SofaCalculator

sofa_calc = SofaCalculator(df)
df = sofa_calc.calculate_all_values()

df.info()

# %% [markdown]
# Hieronder wordt berekend welk percentage van de dataset sepsis heeft en welk percentage niet. Daarbij wordt geen rekening gehouden met het feit dat wanneer bij een patiënt eenmaal sepsis is vastgesteld, deze in de daaropvolgende uren ook sepsis heeft. De berekening gebeurt dus per afzonderlijke rij in de dataset.

# %%
if 'SepsisLabel' not in df.columns:
    raise ValueError("Kolom 'SepsisLabel' ontbreekt in de DataFrame")

counts = df['SepsisLabel'].value_counts().sort_index()
total = len(df)

print('Sepsis verdeling:')
for val, count in counts.items():
    percentage = (count / total) * 100
    print(f'{val}: aantal = {count}, percentage = {percentage:.2f}%')

# %% [markdown]
# ### Intergrate Data
# Deze stap is in dit geval niet nodig, omdat er slechts met één dataset wordt gewerkt. Er hoeven geen andere datasets samengevoegd te worden, waardoor deze handeling geen toegevoegde waarde heeft. Om die reden wordt deze stap in de volgende cyclus overgeslagen.

# %% [markdown]
# ### Format Data
# Voor het trainen van de modellen wordt de dataset eerst gesplitst in een train- en testset. Hierbij bestaat `X` uit alle variabelen behalve het `sepsislabel` en de twee `unit`-variabelen. Het sepsislabel word gebruikt als de y-variabele, terwijl de `unit`-variabelen uitsluitend worden gebruikt om de dataset op een correcte manier te segmenteren.

# %%
from sklearn.model_selection import train_test_split

X = df.drop(columns=['SepsisLabel', 'Unit1', 'Unit2'])
y = df['SepsisLabel'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% [markdown]
# ## Modeling
# Voor de eerste cyclus worden twee verschillende modellen ontwikkeld: een decision tree en een random forest. Het random forest is hierbij een uitbreiding op de decision tree. Vervolgens wordt de prestatie van beide modellen met elkaar vergeleken om te bepalen welk model het beste resultaat oplevert.
# 
# ### Decision Tree
# Dit model wordt getraind op basis van het sepsislabel. Hierdoor kan op basis van de SOFA-scores worden voorspeld of een patiënt in de categorie sepsis valt.

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import dtreeviz

model = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
model.fit(X_train, y_train)

# %% [markdown]
# De prestaties van het model zijn uitzonderlijk goed, maar vereisen nadere controle om te bevestigen of dit correct is.

# %%
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# De visualisatie laat zien hoeveel patiënten in de trainingsdataset op basis van de SOFA-scores als sepsis worden geclassificeerd. Daarnaast zijn er enkele andere variabelen opgenomen, maar deze lijken weinig tot geen invloed te hebben op deze classificatie.

# %%
dt1_model = dtreeviz.model(
    model,
    X_train,
    y_train,
    feature_names=X_train.columns,
    target_name='Sepsis',
    class_names=["No Sepsis", "Sepsis"]
)

dt1_model.view(scale=2)

# %% [markdown]
# ## Random forest
# Net als bij het vorige model wordt ook dit model getraind om het sepsislabel te herkennen. Dit gebeurt op een vergelijkbare manier als bij de decision tree.

# %%
from sklearn.ensemble import RandomForestClassifier

rf1_model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=10,           
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1                
)

rf1_model.fit(X_train, y_train)

# %% [markdown]
# Net zoals bij het voirge lijkt dit model perfect. Dit is heel onwaarschijnlijk en moet dus ook nader na gekekn worden of dit wel correct is.

# %%
y_pred = rf1_model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# Omdat dit model, net als een decision tree, gebruikmaakt van boomstructuren, kijkt het niet naar één enkele boom maar naar meerdere bomen om te bepalen of iemand wel of niet aan het sepsislabel voldoet. Door deze combinatie van meerdere bomen kan het model verschillende patronen meenemen in de beslissing. Daarom lijkt dit model geschikter dan een enkele decision tree, omdat het op basis van meer informatie en perspectieven tot een eindbeslissing komt.

# %%
rf1_tree = rf1_model.estimators_[0]

rf1_tree_model = dtreeviz.model(
    rf1_tree,
    X_train,
    y_train,
    feature_names=X_train.columns,
    target_name='SepsisLabel',
    class_names=["No Sepsis", "Sepsis"]
)

rf1_tree_model.view()

# %% [markdown]
# ### Reflectie
# Zoals bij beide modellen al benoemd, lijken de prestaties te goed. De nauwkeurigheid ligt vrijwel rond de 100%. In de praktijk is dit echter onrealistisch, zeker omdat de dataset nog beperkt is bewerkt en nog niet volledig is opgeschoond. Een belangrijke factor hierin is de aanwezigheid van ontbrekende data. Het gaat hierbij onder andere om kolommen die door de `SofaCalculator` worden gebruikt om de SOFA-scores en bijbehorende sepsislabels te berekenen.
# 
# Op dit moment baseert het model zijn voorspellingen slechts op een beperkt aantal variabelen, zonder rekening te houden met hoe deze waarden daadwerkelijk bijdragen aan het ontstaan van het sepsislabel. Die onderliggende relaties zijn in deze fase nog niet goed meegenomen, onder andere doordat er fouten optraden bij het verwerken en combineren van de relevante tabellen.
# 
# Daarnaast laten evaluatiemethoden zoals het `classification_report` en de `confusion_matrix` uitzonderlijk hoge scores zien. Hoewel dit op het eerste gezicht wijst op een sterk model, is dit waarschijnlijk misleidend. Het is aannemelijk dat er iets mis is gegaan bij het splitsen van de data in trainings- en testsets. Mogelijk is er sprake van overfitting of van datalekken, waarbij de testdata niet goed gescheiden is van de trainingsdata.
# 
# In de volgende iteratie zal de focus daarom liggen op het verbeteren van de datakwaliteit en het correct verwerken van ontbrekende waarden. Ook zullen er extra kolommen worden toegevoegd die nodig zijn voor het berekenen van de qSOFA- en SOFA-scores.
# 
# Hoewel beide modellen opnieuw zullen worden geëvalueerd, ligt de voorkeur op dit moment bij het random forest-model. Dit model is doorgaans beter in staat om complexe patronen te leren en presteert vaak robuuster wanneer de dataset uitgebreider en informatiever wordt.

# %% [markdown]
# # Cycle II
# oals hierboven beschreven, ligt de focus van deze cyclus voornamelijk op het verbeteren van de datakwaliteit en de verwerking ervan. Daarnaast is het belangrijk om de data op te splitsen in een train- en testset. Op die manier wordt beoogd de prestaties en betrouwbaarheid van de modellen te verbeteren.

# %% [markdown]
# ## Voorbereiding
# De eerdere stappen waren grotendeels nog steeds noodzakelijk, en het zou zonde zijn om deze opnieuw uit te voeren. Wat wél opnieuw moet gebeuren, is het meenemen van de overige data die bijdragen aan het bepalen van het `sepsislabel`. Om dit te realiseren, worden eerst de bestaande kolommen die hierop betrekking hebben verwijderd en vervolgens opnieuw berekend. 
# 
# Hier worden eerst de kolommen verwijders om ze opnieuw te berekenen.

# %%
df = df.drop(columns=['qSOFA_partial', 'SOFA_modified_total'])

# %% [markdown]
# Hieronder worden de herberekeningen van deze waarden weergegeven. Dit keer worden de extra kolommen die nodig zijn voor de berekeningen wel in de dataset opgenomen.

# %%
sofa_calc = SofaCalculator(df)

df = sofa_calc.calculate_all_values(True, True)

# %% [markdown]
# Hier is te zien dat er enkele extra kolommen zijn toegevoegd waarmee de SOFA-scores worden berekend. Dit betreft de variabelen `qsofa_resp`, `qsofa_sbp`, `SF_ratio`, `sofa_resp`, `sofa_coag`, `sofa_liver`, `sofa_cv` en `sofa_renal`. Deze waarden vormen de basis voor de punten waaruit de SOFA-scores worden opgebouwd.

# %%
df.info()

# %% [markdown]
# Bij de SF_ratio zijn een aantal afwijkende waarden zichtbaar, waaronder inf en NaN. Deze waarden moeten worden gecorrigeerd of verwerkt, zodat ze bruikbaar zijn in de verdere analyse. De overige waarden lijken wel correct en hoeven niet aangepast te worden.

# %%
df.describe()

# %% [markdown]
# Door het kijken naar deze rows vallen er geen andere dingen op.

# %%
df.head()

# %% [markdown]
# Een aantal kolommen in deze dataset bevat nog ontbrekende waarden (`null`). Deze worden, net als in de vorige cyclus, later vervangen door 0. Dit wordt gedaan om aan te geven dat deze waarden niet zijn gemeten.

# %%
df.isnull().sum()

# %% [markdown]
# Binnen de `SF_ratio` komen opvallend veel `inf`-waarden voor. Dit is waarschijnlijk het gevolg van de manier waarop deze ratio wordt berekend. In de berekening wordt geen rekening gehouden met het voorkomen van `NaN`-waarden, wat ertoe kan leiden dat er ongeldige of oneindige (`inf`) resultaten ontstaan.

# %%
np.isinf(df['SF_ratio']).sum()

# %% [markdown]
# ## Data preperation
# Voordat de data kan worden gebruikt voor de nieuwe modellen, moeten nog enkele kleine aanpassingen worden doorgevoerd. Dit wordt gedaan met als doel om deze keer tot een beter presterend model te komen.
# 
# ### Clean Data 
# Hier worden eerst alle `inf`-waarden vervangen door 0. Vervolgens worden ook alle `null`-waarden aangepast. Dit wordt gedaan om de data consistent te maken met andere niet-gemeten waarden.

# %%
df['SF_ratio'] = df['SF_ratio'].replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# %% [markdown]
# ### Construct Data
# Het enige dat hier aan de dataset wordt toegevoegd is `Sepsis_Future`. Deze kolom geeft aan of iemand, op basis van de SOFA-waardes, zes uur later sepsis heeft of niet. Hiermee kan worden geanalyseerd of iemand volgens de gekozen criteria sepsis ontwikkelt en hoe deze verdeling eruitziet.
# 
# Om dit te realiseren wordt de dataset eerst gesorteerd op `Patient_ID` en `Hour`, waarbij `Hour` aangeeft hoeveel uur een patiënt al in het ziekenhuis is opgenomen.

# %%
df = df.sort_values(['Patient_ID', 'Hour'])
df['Sepsis_Future'] = df.groupby('Patient_ID')['SepsisLabel'].shift(-6)

# %% [markdown]
# Over deze stap is nog niet volledig duidelijk of deze noodzakelijk is. Hierbij worden alle rijen verwijderd waarvoor geen `Sepsis_Future`-waarde beschikbaar is. Voor het opschonen van de dataset, voorafgaand aan het trainen van het model, lijkt dit echter wel een zinvolle stap. Aangezien er geen data aanwezig is voor sommige patienten na een `X` aantal uren. Dit word dan automatisch als `NaN` gezet.

# %%
df = df.dropna(subset=['Sepsis_Future'])

# %% [markdown]
# ### Format Data
# In de vorige cyclus waren de modellen relatief eenvoudig opgezet. Daarbij werd direct gebruikgemaakt van de `SOFA`- en `qSOFA`-scores om te bepalen of een patiënt sepsis had. Omdat deze scores zelf al gebaseerd zijn op klinische criteria voor sepsis, resulteerde dit in ogenschijnlijk perfecte voorspellingen met een nauwkeurigheid van 100%. Hoewel dit op het eerste gezicht positief lijkt, geeft het geen realistisch beeld van de daadwerkelijke voorspellende kracht van het model.
# 
# Daarom wordt in deze cyclus een andere aanpak gehanteerd. De samengestelde `SOFA`- en `qSOFA`-scores worden bewust buiten beschouwing gelaten. In plaats daarvan wordt uitsluitend gewerkt met de onderliggende medische parameters waaruit deze scores zijn opgebouwd. Op basis van deze ruwe medische gegevens wordt vervolgens bepaald of een patiënt sepsis heeft. Het doel hiervan is om een robuuster en realistischer voorspellingsmodel te ontwikkelen dat minder afhankelijk is van reeds afgeleide klinische indicatoren.
# 
# Daarnaast verschuift de focus van het model. Waar in de vorige cyclus voornamelijk werd gekeken naar het vaststellen of een patiënt op dat moment sepsis had, ligt de nadruk nu op het voorspellen of een patiënt in de toekomst sepsis zal ontwikkelen. Hiermee wordt het model meer gericht op vroegtijdige signalering en preventieve ondersteuning.
# 
# Ook de manier waarop de data wordt opgesplitst, is in deze cyclus verbeterd. In de vorige cyclus werd de dataset willekeurig verdeeld over de `train`- en `test`-set. Hierdoor konden gegevens van dezelfde patiënt in beide datasets terechtkomen. Dit is onwenselijk, omdat het leidt tot datalekken en een vertekend beeld geeft van de prestaties van het model.
# 
# Om dit probleem op te lossen, wordt de dataset nu gesplitst op basis van unieke `Patient_ID`s. Hierdoor blijven alle gegevens van een individuele patiënt volledig binnen één dataset, namelijk óf de `train`-set óf de `test`-set. Dit zorgt voor een realistischer evaluatieproces en betrouwbaardere resultaten.
# 
# Voor deze werkwijze is de helperfunctie `train_test_split_by_patient` ontwikkeld. Deze functie zorgt ervoor dat de dataset op een consistente en correcte manier wordt opgesplitst. Daarnaast verhoogt dit de herbruikbaarheid van de code, zodat dezelfde logica bij toekomstige iteraties of wijzigingen in de dataset eenvoudig opnieuw kan worden toegepast. Hiermee wordt voorkomen dat identieke bewerkingen telkens opnieuw moeten worden uitgevoerd of dat dezelfde code meerdere keren herschreven moet worden

# %%
from helpers.notebook_helpers import train_test_split_by_patient

train_patients, test_patients = train_test_split_by_patient(df)

# %% [markdown]
# Hieronder worden de `train`- en `test`-sets opgehaald op basis van de huidige dataset. Voor deze voorspelling is `Patient_ID` overbodig en word deze daarom nog verwijderd.

# %%
from helpers.notebook_helpers import get_train_test_data_by_patient

X_train, y_train, train_patient_ids, X_test, y_test, test_patient_ids = get_train_test_data_by_patient(
    original_df=df, 
    train_patients=train_patients, 
    test_patients=test_patients,
    y_target='Sepsis_Future', 
    delete_patient_ids=True
)

# %% [markdown]
# ## Modeling
# In dit hoofdstuk worden dezelfde modellen op identieke wijze opgebouwd als in de vorige cyclus. Het verschil is dat er ditmaal gebruik wordt gemaakt van een beter voorbereide dataset. In hoeverre dit daadwerkelijk tot betere prestaties leidt, zal blijken uit de modelresultaten, al is de verwachting dat de verbeterde datakwaliteit hier een positieve bijdrage aan levert.
# 
# ### Decision Tree
# Hieronder wordt de opbouw van de nieuwe decision tree weergegeven. Hierbij ligt de focus met name op de `classification report` en de `confusion matrix`, om te beoordelen of het model daadwerkelijk is verbeterd. Het model wordt getraind en geëvalueerd met de eerder samengestelde `train`- en `test`-datasets.

# %%
model = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
model.fit(X_train, y_train)

# %% [markdown]
# Uit de resultaten blijkt dat het model minder goed presteert dan in de vorige cyclus. Met name de lage precision voor klasse 1 en de relatief hoge hoeveelheid fout-positieve voorspellingen laten zien dat het model moeite heeft om sepsis nauwkeurig te identificeren. Tegelijkertijd is de recall voor klasse 1 wel hoog, wat betekent dat een groot deel van de daadwerkelijke gevallen wordt herkend, maar ten koste van veel onterechte voorspellingen.
# 
# Hoewel de prestaties op het eerste gezicht slechter lijken, is dit juist een positieve ontwikkeling. In de vorige cyclus was er sprake van datalekken en een onjuiste datasplitsing, waardoor de resultaten te optimistisch waren. In de huidige aanpak is de data correct verdeeld op basis van unieke patiënten, wat zorgt voor een realistischer en betrouwbaarder beeld van de modelprestaties.

# %%
y_pred_dt2 = model.predict(X_test)

print(classification_report(y_test, y_pred_dt2))
print(confusion_matrix(y_test, y_pred_dt2))

# %% [markdown]
# Kijkend naar het model valt op dat het slechts een beperkt aantal kenmerken gebruikt om te bepalen of iemand sepsis heeft of niet. Hierdoor lijkt het model onvoldoende complex en daarmee minder geschikt voor deze vraagstelling. Tegelijkertijd is het wel positief om te zien dat, ondanks deze beperkingen, er sprake is van een verbetering ten opzichte van de vorige aanpak.

# %%
dt2_model = dtreeviz.model(
    model,
    X_train,
    y_train,
    feature_names=X_train.columns,
    target_name='Sepsis',
    class_names=["No Sepsis", "Sepsis"]
)

dt2_model.view(scale=2)

# %% [markdown]
# Hieronder worden de hulpmethoden geïmporteerd die verantwoordelijk zijn voor het exporteren van de prediction set en het berekenen van de bijbehorende score. Omdat deze functionaliteit regelmatig wordt gebruikt, is hiervoor een aparte helpermethode ontwikkeld om de code herbruikbaar en overzichtelijk te houden.

# %%
from helpers.notebook_helpers import export_prediction_set, print_utiltiy_score

export_prediction_set(test_patient_ids, df, y_pred_dt2)
print_utiltiy_score()

# %%
from scepsis_prediction.evaluation import evaluate_sepsis_score

utility = evaluate_sepsis_score(
    "testset (with label).csv",
    "predictions.csv"
)

print(utility)

# %% [markdown]
# ### Random Forest
# Hetzelfde geldt hier. Het model wordt op dezelfde manier opgebouwd als het vorige, maar ditmaal met gebruik van de nieuwe `train`- en `test`datasets.

# %%
rf2_model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=3,           
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1                
)

rf2_model.fit(X_train, y_train)

# %% [markdown]
# Uit de resultaten van het random forest blijkt dat het model beter presteert dan de decision tree. De accuracy ligt hoger en ook de verdeling in de confusion matrix laat zien dat er minder fout-positieve voorspellingen worden gedaan. Dit wijst erop dat het model beter in staat is om onderscheid te maken tussen de klassen.
# 
# Tegelijkertijd blijft de precision voor klasse 1 laag, wat betekent dat een groot deel van de voorspelde sepsisgevallen onterecht is. De recall voor klasse 1 is daarentegen nog steeds redelijk hoog, waardoor het model wel een groot deel van de daadwerkelijke gevallen weet te identificeren.
# 
# Net als bij het vorige model geldt dat de prestaties realistischer zijn dan in de eerdere cyclus. Door de verbeterde datasplitsing op basis van unieke patiënten geeft dit model een betrouwbaarder beeld van de werkelijke prestaties, ook al zijn deze minder optimaal dan voorheen.

# %%
y_pred_rf2 = rf2_model.predict(X_test)

print(classification_report(y_test, y_pred_rf2))
print(confusion_matrix(y_test, y_pred_rf2))

# %% [markdown]
# Uit deze visualisatie blijkt dat het model rekening houdt met een veel meer variabelen uit de dataset. In plaats van slechts enkele kenmerken, worden meerdere factoren meegenomen die gezamenlijk bijdragen aan de classificatie of iemand wel of geen sepsis heeft.
# 
# Daarnaast valt op dat ook variabelen zoals `hour` worden meegenomen. Dit geeft waardevol inzicht in het tijdsaspect van de metingen en laat zien op welke momenten bepaalde waarden een rol spelen bij het voorspellen van sepsis. Hierdoor wordt er als het ware per uur gekeken naar de kans dat een patiënt de aandoening heeft of zal ontwikkelen.

# %%
rf2_tree = rf2_model.estimators_[0]

rf2_tree_model = dtreeviz.model(
    rf2_tree,
    X_train,
    y_train,
    feature_names=X_train.columns,
    target_name='Sepsis',
    class_names=["No Sepsis", "Sepsis"]
)

rf2_tree_model.view(scale=2)

# %%
export_prediction_set(test_patient_ids, df, y_pred_rf2)
print_utiltiy_score()

# %% [markdown]
# ## Reflectie
# Uit deze cyclus blijkt dat de modellen een duidelijk realistischer beeld geven van de prestaties dan in de vorige cyclus. Hoewel de resultaten in theorie slechter lijken, is dit juist een positieve ontwikkeling. In de eerdere cyclus was er sprake van overfitting en datalekken door een onjuiste datasplitsing, waardoor de prestaties te optimistisch werden weergegeven. In de huidige aanpak, waarbij is gesplitst op basis van unieke patiënten, zijn de resultaten betrouwbaarder en beter representatief voor de praktijk.
# 
# De decision tree laat zien dat het model moeite heeft om sepsis nauwkeurig te classificeren. Met name de lage precision voor klasse 1 en het hoge aantal fout-positieve voorspellingen vallen op, ondanks een relatief hoge recall. Daarnaast gebruikt het model slechts een beperkt aantal kenmerken, wat erop wijst dat het te eenvoudig is voor dit complexe vraagstuk. Om die reden wordt dit model niet verder meegenomen in volgende cycli.
# 
# Het random forest presteert aantoonbaar beter dan de decision tree. De accuracy ligt hoger en de verdeling in de confusion matrix is evenwichtiger, met minder fout-positieve voorspellingen. Dit duidt erop dat het model beter onderscheid kan maken tussen de klassen. Tegelijkertijd blijft de precision voor klasse 1 laag, wat aangeeft dat er nog steeds veel onterechte positieve voorspellingen worden gedaan. De recall blijft daarentegen redelijk hoog, waardoor het model wel een groot deel van de daadwerkelijke sepsisgevallen weet te identificeren.
# 
# Uit de bijbehorende visualisatie blijkt bovendien dat het random forest gebruikmaakt van een meer variabelen. In tegenstelling tot de decision tree worden meerdere kenmerken meegenomen in de besluitvorming, waaronder ook tijdsgerelateerde variabelen zoals `hour`. Dit biedt extra inzicht in het moment waarop bepaalde metingen bijdragen aan de voorspelling en maakt het model beter geschikt voor dit type probleem.
# 
# Voor de volgende cyclus ligt de focus op het verkennen van alternatieve modellen, zoals een `AMIRA`- of `SAMIRA`-model. In een latere cyclus kan vervolgens worden gekeken naar het verfijnen van deze modellen en het vergelijken van de best presterende variant met het huidige random forest.

# %% [markdown]
# # Cycle III
# In deze cyclus ligt de focus op het ontwikkelen van modellen waarbij het tijdsverloop van een patiënt, in combinatie met de medische waarden gedurende het ziekenhuisverblijf, centraal staat. Door de gegevens per uur te analyseren, wordt gehoopt waardevolle en bruikbare inzichten te verkrijgen die kunnen bijdragen aan het beter voorspellen van sepsis.

# %% [markdown]
# ## Voorbereiding
# Voor deze cyclus is geen aanvullende voorbereiding van de data nodig. Er wordt gebruikgemaakt van de reeds geprepareerde dataset uit de vorige cyclus. Alleen de `train`- en `test`-split wordt opnieuw uitgevoerd.

# %% [markdown]
# ## Data preperation
# ### Clean Data
# Deze stap word in deze cycle overgeslagen omdat er verder word gewerkt met de data van de vorige cyclus. 

# %% [markdown]
# ### Construct Data
# Hieronder staat een kleine helpermethode die de duur van een bezoek berekent, oftewel het tijdsverschil (tijdsdeltа) van een opname. In de dataset begint de tijdsregistratie van een bezoek bij 0. Daarom wordt hier de maximale waarde genomen en met 1 verhoogd. Dit geeft de totale duur weer dat een patiënt in het ziekenhuis heeft verbleven. Door deze berekening toe te passen, wordt het eenvoudiger om de `ARIMA`- en `SARIMA`-model verder uit te werken.
# 
# Deze functie geeft een `Series` terug oftewel een kolom. Dit omdat binnen de functie gefilterd word op bepaalde data. Daarom is het makkelijker om deze terug tegeven inplaats van een geheel `DataFrame`.
# 
# Omdat in de functie `get_train_test_data_by_patient` de `Patient_ID` wordt verwijderd, moeten de basisvoorbereidingen opnieuw worden uitgevoerd. Dit is noodzakelijk omdat `calculateVisitTime` wordt berekend op basis van zowel de `Patient_ID` als de `Hour`. `Patient_ID` is namelijk in de vorige cycle uit de `train`- en `test`-data verwijderd. Daarom word hier `Patient_ID` achteraf verwijderd.

# %%
from helpers.notebook_helpers import calculate_visit_time

train_patients, test_patients = train_test_split_by_patient(df)

X_train, y_train, _, X_test, y_test, _ = get_train_test_data_by_patient(
    original_df=df, 
    train_patients=train_patients, 
    test_patients=test_patients,
    y_target='Sepsis_Future',
)

X_train['visit_duration'] = calculate_visit_time(X_train)
X_test['visit_duration'] = calculate_visit_time(X_test)

X_train = X_train.drop(columns=['Patient_ID'])
X_test = X_test.drop(columns=['Patient_ID'])

# %% [markdown]
# ### Format Data
# Deze stap word in deze cycle overgeslagen omdat er verder word gewerkt met de data van de vorige cyclus. 

# %% [markdown]
# ## Modeling
# In deze stap worden twee verschillende modellen ontwikkeld, namelijk `AMIRA` en `SARIMA`. Het doel hiervan is om meer inzicht te krijgen in het verloop van een ziekenhuisbezoek op uurbasis.
# 
# ### AMIRA

# %%
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA

# %% [markdown]
# ### SAMIRA

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %% [markdown]
# ## Reflectie
# Ondanks de pogingen is het niet gelukt om één van beide modellen succesvol werkend te krijgen. Voor zowel `AMIRA` als `SARIMA` bleek het niet mogelijk om alle benodigde features correct te integreren, terwijl deze juist cruciaal zijn voor het voorspellen van sepsis.
# 
# Een mogelijke workaround zou zijn om voor elke feature afzonderlijke modellen te trainen en deze vervolgens te combineren. Deze aanpak is echter zeer tijdsintensief, waardoor de investering in verhouding tot de opbrengst niet efficiënt is. Daarom is ervoor gekozen om deze richting niet verder te verkennen en de tijd te besteden aan alternatieve modellen.
# 
# In de volgende cyclus zal daarom worden gekeken naar verschillende boosting-modellen, met als doel om via deze methode een betere balans te vinden tussen modelprestatie, complexiteit en praktische toepasbaarheid.
# 
# # Cycle IV
# In deze cyclus wordt gekeken naar verschillende boosting-modellen. In dit geval gaat het om twee relatief eenvoudige modellen: `Gradient Boosting` en `XGBoost`. Het doel hiervan is om te onderzoeken of hiermee een goed presterend model kan worden ontwikkeld.

# %% [markdown]
# ## Voorbereiding
# Qua voorbereiding is er weinig extra nodig. De dataset uit de vorige cycli is voldoende om hierop verder te bouwen, waardoor de volledige data-preparatiestap wordt overgeslagen. Het enige dat nog moet gebeuren, is het verwijderen van de `visit_duration`-variabele uit de vorige cyclus.

# %%
X_train = X_train.drop(columns=['visit_duration'])
X_test = X_test.drop(columns=['visit_duration'])

y_train = y_train.drop(columns=['visit_duration'])
y_test = y_test.drop(columns=['visit_duration'])

# %% [markdown]
# ## Modeling
# ### Gradient boosting

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb1_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb1_model.fit(X_train, y_train)

# %%
y_pred_gb1 = gb1_model.predict(X_test)

print(classification_report(y_test, y_pred_gb1))
print(confusion_matrix(y_test, y_pred_gb1))

# %%
export_prediction_set(test_patient_ids, df, y_pred_gb1)
print_utiltiy_score()

# %% [markdown]
# ### Xgboost

# %%
from xgboost import XGBClassifier

xgb1_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb1_model.fit(X_train, y_train)

# %%
y_pred_xgb1 = xgb1_model.predict(X_test)

print(classification_report(y_test, y_pred_xgb1))
print(confusion_matrix(y_test, y_pred_xgb1))

# %%
export_prediction_set(test_patient_ids, df, y_pred_xgb1)
print_utiltiy_score()

# %% [markdown]
# ### Feature Importance – Cycle IV
# Om beter te begrijpen welke variabelen de modellen het meest beïnvloeden, worden de feature importances van zowel het Gradient Boosting- als het XGBoost-model gevisualiseerd. Dit geeft inzicht in welke medische metingen het meest bijdragen aan de voorspelling van sepsis in dit stadium van de analyse.

# %%
def plot_feature_importance(model, feature_names, title, color='steelblue', top_n=20):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.sort_values().plot(kind='barh', ax=ax, color=color)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Importance (impurity-based)')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.show()

# Gradient Boosting – Cycle IV
plot_feature_importance(
    gb1_model,
    X_train.columns,
    'Top 20 Feature Importances – Gradient Boosting (Cycle IV)',
    color='steelblue'
)

# XGBoost – Cycle IV
plot_feature_importance(
    xgb1_model,
    X_train.columns,
    'Top 20 Feature Importances – XGBoost (Cycle IV)',
    color='darkorange'
)

# %% [markdown]
# ## Reflectie
# Beide modellen laten betere prestaties zien dan de random forest uit de vorige cycli. Tegelijkertijd bestaat er echter een vermoeden van mogelijke datalekage, omdat alle waarden van Sepsis_Future op 0 lijken te zijn gezet. Dit kan de resultaten mogelijk vertekenen.
# 
# Daarnaast kan er nog winst behaald worden door verder te kijken naar feature engineering om het model te verbeteren en robuuster te maken.

# %% [markdown]
# # Cycle V
# In deze cycle wil ik de focus leggen op het opnieuw verwerken/resetten van de dataset. Dit voornamelijk om de missende waardes beter bij te vullen. Hierdoor hoop ik dat de medische waardes die ik bereken voor de `SOFA`-scores ook wat meer voorstellen dam dat ze nu doen. Grotendeels van de aanpassingen en toevoegingnen ik aan de dataset ga hier wel herhaald worden. 
# 
# ## Voorbereiding
# Hier word eerst de data opnieuw ingelezen en wat andere simpele stappen die in voorgaande cycles ook is uitgevoerd.

# %%
from helpers.notebook_helpers import prep_dataset

df = read_dataset()
df = prep_dataset(df)

# %% [markdown]
# ## Data preperation
# Deze stap is hier overbodig, omdat in de voorbereiding al gebruik wordt gemaakt van de functie `prep_dataset`. Deze functie prepareert de dataset volledig, zodat deze direct gebruikt kan worden zonder telkens aanvullende stappen uit te voeren of dezelfde code opnieuw te schrijven. 
# 
# ### Clean Data
# Deze stap word hier overgeslagen. 

# %% [markdown]
# ### Construct Data
# Voor de feature engineering is een aparte klasse ontwikkeld. Dit is gedaan om het notebook overzichtelijk en beter onderhoudbaar te houden. Binnen deze klasse worden verschillende afgeleide medische kenmerken berekend, met als doel het aantal bruikbare features uit te breiden en daarmee mogelijk de utility score van de modellen te verbeteren.
# 
# De klasse bevat diverse functies waarmee extra waarden aan de dataset kunnen worden toegevoegd. In deze cyclus wordt uitsluitend gebruikgemaakt van de functie `add_all_features`. Zoals de naam al suggereert, voegt deze functie automatisch alle berekenbare features toe aan de dataset.
# 
# Bij het aanroepen van deze functie zijn vier verschillende configuraties mogelijk:
# - Alle features toevoegen, inclusief rolling- en temporal-features
# - Alle features toevoegen inclusief temporal-features, maar zonder rolling-features
# - Alle features toevoegen inclusief rolling-features, maar zonder temporal-features
# - Alleen de basisfeatures toevoegen, zonder rolling- en temporal-features
# 
# Er bestond namelijk twijfel over de daadwerkelijke meerwaarde van rolling- en temporal-features voor de prestaties en utility score van de modellen. De enige manier om dit betrouwbaar vast te stellen, is door de verschillende configuraties daadwerkelijk te testen en met elkaar te vergelijken.
# 
# Daarnaast wordt er gekozen voor zowel een forward-fill als een backward-fill om ontbrekende waarden (`null`-waarden) in de dataset op te vullen. Dit is noodzakelijk omdat sommige modellen niet kunnen worden uitgevoerd wanneer de dataset ontbrekende waarden bevat. Hierdoor wordt de dataset volledig bruikbaar gemaakt voor verdere modeltraining en evaluatie.

# %%
from scepsis_prediction.feature_engineering import add_all_features

df = add_all_features(df, include_rolling=True, include_temporal=True)
df = df.ffill().bfill()

# %% [markdown]
# ### Format Data

# %%
X_train, y_train, train_patient_ids, X_test, y_test, test_patient_ids = get_train_test_data_by_patient(
    original_df=df, 
    train_patients=train_patients, 
    test_patients=test_patients, 
    y_target='Sepsis_Future',
    delete_patient_ids=True
)

# %% [markdown]
# ## Modeling
# 
# ### Random forest

# %%
rf5_model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=3,           
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1                
)

rf5_model.fit(X_train, y_train)

# %%
y_pred_rf5 = rf5_model.predict(X_test)

print(classification_report(y_test, y_pred_rf5))
print(confusion_matrix(y_test, y_pred_rf5))

# %%
rf5_tree = rf5_model.estimators_[0]

rf5_tree_model = dtreeviz.model(
    rf5_tree,
    X_train,
    y_train,
    feature_names=X_train.columns,
    target_name='SepsisLabel',
    class_names=["No Sepsis", "Sepsis"]
)

rf5_tree_model.view(scale=2)

# %%
export_prediction_set(test_patient_ids, df, y_pred_rf5)
print_utiltiy_score()

# %% [markdown]
# ### Gradient boosting

# %%
gb5_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb5_model.fit(X_train, y_train)

# %%
y_pred_gb5 = gb5_model.predict(X_test)

print(classification_report(y_test, y_pred_gb5))
print(confusion_matrix(y_test, y_pred_gb5))

# %%
export_prediction_set(test_patient_ids, df, y_pred_gb5)
print_utiltiy_score()

# %% [markdown]
# ### Xgboost

# %%
xgb5_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb5_model.fit(X_train, y_train)

# %%
y_pred_xgb5 = xgb5_model.predict(X_test)

print(classification_report(y_test, y_pred_xgb5))
print(confusion_matrix(y_test, y_pred_xgb5))

# %%
export_prediction_set(test_patient_ids, df, y_pred_xgb5)
print_utiltiy_score()

# %% [markdown]
# ### Feature Importance – Cycle V
# Met de uitgebreidere feature-set (inclusief rolling- en temporal-features) uit Cycle V zijn er meer variabelen beschikbaar. De onderstaande visualisaties laten zien welke features de boosting-modellen het zwaarst laten meewegen. De vergelijking met Cycle IV geeft inzicht in de toegevoegde waarde van de extra afgeleide kenmerken.

# %%
# Gradient Boosting – Cycle V
plot_feature_importance(
    gb5_model,
    X_train.columns,
    'Top 20 Feature Importances – Gradient Boosting (Cycle V)',
    color='seagreen'
)

# XGBoost – Cycle V
plot_feature_importance(
    xgb5_model,
    X_train.columns,
    'Top 20 Feature Importances – XGBoost (Cycle V)',
    color='tomato'
)

# %% [markdown]
# ## Reflectie
# Deze modellen zijn meerdere keren getest in uiteenlopende configuraties en met verschillende instellingen. Na verloop van tijd werd het echter lastig om nog overzicht te houden over welke aanpassingen precies verantwoordelijk waren voor bepaalde resultaten en scores. Hierdoor ontstond een onoverzichtelijk proces dat moeilijk reproduceerbaar werd.
# 
# Daarom zal in de volgende cyclus de focus verschuiven naar modeloptimalisatie. Door gebruik te maken van geautomatiseerde optimalisatietechnieken kan systematisch worden gezocht naar de best presterende configuraties, zonder dat dit handmatig hoeft te gebeuren. Op deze manier kan efficiënter het optimale model worden gevonden, waarna dit vervolgens gevalideerd kan worden.
# 
# Uit de uitgevoerde experimenten bleek daarnaast dat `Random Forest` in de meeste gevallen de minst goede resultaten opleverde ten opzichte van de andere geteste modellen.
# 
# # Cycle VI

# %% [markdown]
# Deze cyclus legt voornamelijk de focus op het optimaliseren van de modellen. Daarnaast is er in de tussentijd nog een extra boosting-model onderzocht, namelijk CatBoost. Dit model staat bekend om het leveren van sterke en stabiele prestaties en zou in veel gevallen betere resultaten kunnen behalen dan andere veelgebruikte boosting-modellen. Daarom wordt ook dit model meegenomen in de experimenten, met als doel de prestaties van de voorspellingen verder te verbeteren.
# 
# ### Optimalisatie
# Om dit proces uit te voeren zijn twee losse Python-scripts ontwikkeld die de modeloptimalisatie automatiseren: `optimalisation.py` en `get_best_utility_score.py`.
# 
# Het script `optimalisation.py` voert per model meerdere optimalisatietrials uit voor de verschillende feature sets die in de vorige cyclus zijn benoemd. Hierbij worden diverse configuraties getest om te bepalen welke combinatie van modelinstellingen en feature sets de beste prestaties oplevert.
# 
# Tijdens de eerste optimalisatieronde ligt de focus op het identificeren van de best presterende modellen en de meest effectieve feature-setcombinaties. De resultaten van deze experimenten worden opgeslagen in `.csv`-bestanden.
# 
# Vervolgens wordt het script `get_best_utility_score.py` gebruikt om automatisch de modellen met de hoogste utility scores uit deze resultaten te selecteren. Hierdoor ontstaat een overzichtelijk en reproduceerbaar proces voor het vergelijken en evalueren van de verschillende modellen en configuraties.
# 
# #### Ronde 1
# In de eerste rondes worden alle modellen systematisch met elkaar vergeleken over de verschillende beschikbare feature-sets. Hierdoor kan worden vastgesteld welke combinaties van modellen en feature-sets de beste prestaties opleveren.

# %%
results_df = pd.read_csv('optuna_storage/results_10-05-2026.csv', sep=',')

# %%
results_df.sort_values(by=['model', 'f1'])
results_df.head(15)

# %% [markdown]
# Uit de eerste optimalisatierondes kwamen een aantal duidelijke conclusies naar voren.
# 
# **Verwijderde feature-sets**
# De volgende feature-sets lieten over het algemeen de slechtste prestaties zien wanneer gekeken werd naar de best presterende runs:
# - All
# - Temporal_only
# Daarom zijn deze feature-sets in de vervolgexperimenten niet verder meegenomen.
# 
# **Verwijderd model**
# Ook het model `XGB` liet over het algemeen de minst goede prestaties zien in vergelijking met de andere modellen. Om die reden is besloten dit model eveneens uit de volgende optimalisatierondes te verwijderen.
# 
# **Rekentijd van CatBoost**
# Hoewel `CatBoost` goede resultaten liet zien, bleek een groot nadeel de aanzienlijke rekentijd te zijn. Het trainen van dit model duurde ongeveer drie tot vier keer langer dan de overige modellen. Waar de andere modellen gemiddeld ongeveer 40 minuten nodig hadden voor 50 trials, liep dit bij `CatBoost` op tot meerdere uren.
# 
# Vanwege deze lange uitvoeringstijd moesten de experimenten zelfs gedurende de hele avond blijven draaien. De extra rekentijd bleek uiteindelijk niet in verhouding te staan tot de winst in prestaties. Daarom is besloten om `CatBoost` in de volgende optimalisatierondes niet meer mee te nemen.

# %% [markdown]
# #### Ronde 2
# In deze ronde is verder gewerkt met de modellen uit de vorige optimalisatieronde. Omdat `CatBoost` vanwege de lange rekentijd is afgevallen, is ervoor gekozen om alsnog gebruik te maken van `XGB`, ondanks dat de resultaten hiervan eerder relatief matig waren.
# 
# Daarnaast ontstond het vermoeden dat er in de voorgaande rondes en cycles nog steeds sprake was van data leakage. In eerdere experimenten werd namelijk eerst een `forward-fill` toegepast, gevolgd door een `backward-fill`. Vooral deze laatste stap bleek waarschijnlijk problematisch, omdat hiermee informatie uit toekomstige meetmomenten onbedoeld gebruikt kon worden bij eerdere observaties.
# 
# Om dit risico te minimaliseren, is besloten de modellen opnieuw te trainen met uitsluitend de twee best presterende feature-sets. In plaats van ontbrekende waarden op te vullen met een `backward-fill`, worden de `null`-waarden nu volledig uit de dataset verwijderd. Hiermee blijft de integriteit van de data beter behouden en wordt gehoopt betrouwbaardere modelprestaties te behalen.
# 
# Hieronder wordt een extra import toegevoegd waarmee de door `Optuna` opgeslagen modellen kunnen worden ingeladen. Hierdoor kan vervolgens de juiste `X_test`-dataset aan het model worden meegegeven om voorspellingen (`y_pred`) te genereren voor de validatie van het model.

# %%
import joblib

# %%
df = read_dataset()
df = prep_dataset(df, add_sepsis_future=True)
df = add_all_features(df=df, include_temporal=False, include_rolling=True)

df = df.ffill()
df = df.dropna()

train_patients, test_patients = train_test_split_by_patient(df)
_, _, _, X_test, _, test_patient_ids = get_train_test_data_by_patient(
    original_df=df,
    train_patients=train_patients,
    test_patients=test_patients,
    y_target='Sepsis_Future',
    delete_patient_ids=True,
)

# %% [markdown]
# Hier is gekozen voor het `rolling_only_xgb`-model, niet omdat dit de hoogste F1-score behaalt, maar omdat het een zeer hoge recall heeft. Dit is belangrijk om alarmvermoeidheid in de klinische praktijk te helpen verminderen.
# 
# Daarnaast is er een resultatenoverzicht met meerdere modellen, gebaseerd op een mix van alle uitgevoerde tests van `10-05-2026`. Hierdoor zijn niet alle resultaten volledig onderling vergelijkbaar, omdat de modellen niet allemaal zijn getraind met dezelfde manier van dataverdeling en feature engineering.

# %%
loaded = joblib.load('optuna_storage/saved_models/rolling_only_xgb_10-05-2026.pkl')

model = loaded["model"]
threshold = loaded.get("threshold", 0.5)

proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

# %% [markdown]
# Dit model presteert zeer goed, mogelijk zelfs te goed. Daardoor bestaat het vermoeden dat er nog steeds sprake is van een vorm van data leakage. Een waarschijnlijke oorzaak hiervan is het gebruik van `Sepsis_Future` als `y_target`. Deze variabele is immers gebaseerd op een gebeurtenis die pas zes rijen (zes uur) later plaatsvindt, waardoor er onbedoeld informatie uit de toekomst in het model kan zijn terechtgekomen.
# 
# Om dit te voorkomen, lijkt het verstandiger om in de volgende ronde terug te vallen op `SepsisLabel` als targetvariabele. Dit zorgt voor een meer realistische en striktere scheiding tussen input en doelvariabele, wat de betrouwbaarheid van de resultaten ten goede zou moeten komen.

# %%
export_prediction_set(test_patient_ids, df, y_pred)
print_utiltiy_score()

# %% [markdown]
# #### Ronde 3
# Dit is naar verwachting de laatste ronde van de optimalisatie. Zoals hierboven benoemd worden de best presterende modellen opnieuw getraind, ditmaal met `SepsisLabel` als `y_target`. Hiermee wordt geprobeerd het mogelijke probleem van data leakage te verhelpen en te komen tot een betrouwbaarder en daadwerkelijk bruikbaar model.

# %%
df = read_dataset()
df = prep_dataset(df, add_sepsis_future=False)
df = add_all_features(df=df, include_temporal=True, include_rolling=True)

df = df.ffill()
df = df.dropna()

train_patients, test_patients = train_test_split_by_patient(df)
_, _, _, X_test, _, test_patient_ids = get_train_test_data_by_patient(
    df,
    train_patients,
    test_patients,
    y_target='SepsisLabel',
    delete_patient_ids=True,
)

# %% [markdown]
# Hier is gekozen voor het `all_lgbm`-model. Dit model behaalt een hoge `F1`-score en laat daarnaast, in verhouding tot deze score, ook een hoge `recall` zien. Dit is met name belangrijk in de context van het verminderen van alarmvermoeidheid in de klinische praktijk, omdat een goede balans tussen nauwkeurigheid en het beperken van onnodige alarmsignalen essentieel is.

# %%
results_df = pd.read_csv('optuna_storage/results_13-05-2026.csv', sep=',')
results_df.sort_values(by=['model', 'f1'])
results_df.head()

# %%
loaded = joblib.load('optuna_storage/saved_models/all_lgbm_13-05-2026.pkl')

model = loaded["model"]

# %%
threshold = loaded.get("threshold", 0.5)

proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

# %% [markdown]
# Hier is een veel realistischer utility score te zien van ongeveer `0.138` (afgerond). Hoewel deze waarde niet hoog is, is dat juist een positief teken. Dit suggereert dat er geen sprake meer is van data leakage en dat het model een eerlijker en betrouwbaarder beeld geeft van de werkelijke prestaties. Hiermee is er uiteindelijk een degelijk model gevormd.

# %%
export_prediction_set(test_patient_ids, df, y_pred)
print_utiltiy_score()

# %%
# Feature Importance
import lightgbm as lgb

booster = model.booster_
fi_gain = pd.Series(
    booster.feature_importance(importance_type='gain'),
    index=booster.feature_name()
).sort_values(ascending=False)

top20 = fi_gain.head(20).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
top20.plot(kind='barh', ax=ax, color='mediumslateblue')
ax.set_title('Top 20 Feature Importances – all_lgbm (gain)', fontweight='bold')
ax.set_xlabel('Importance (gain)')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.show()

print('Top 10 meest invloedrijke features:')
print(fi_gain.head(10).to_string())

# %% [markdown]
# ### Reflectie
# Dit is tot nu toe de beste en enige cyclus waarin daadwerkelijk een bruikbaar model is ontstaan. Tegelijkertijd is het jammer dat deze cyclus zoveel tijd heeft gekost, waardoor er minder ruimte overbleef voor verdere feature engineering en experimentatie.
# 
# In de eerste paar runs die ’s nachts waren gestart, traden onverwachte crashes op. Dit kwam doordat de code aanvankelijk in het notebook draaide en later zonder duidelijke aanleiding is vastgelopen. Het nadeel hiervan is dat dit pas de volgende ochtend zichtbaar werd, waardoor er kostbare tijd verloren ging. Nadat dit probleem is verholpen, verliep het proces beter.
# 
# Toch zorgden de aanhoudende data leakage-issues ervoor dat het hertrainen van modellen telkens opnieuw moest gebeuren, wat het proces aanzienlijk vertraagde. Bovendien lag het werk grotendeels stil tijdens deze runs, omdat gewacht moest worden tot alle experimenten waren afgerond voordat het beste model kon worden geselecteerd.
# 
# Alles bij elkaar genomen is het daarom teleurstellend dat er in deze cyclus minder tijd beschikbaar was voor feature engineering, terwijl dit juist een belangrijke stap had kunnen zijn om een nog beter presterend model te ontwikkelen.

# %% [markdown]
# # Conclusie
# ## Terugblik op de vraagstelling
# De centrale onderzoeksvraag luidde: *Hoe kan Isala ziekenhuis met behulp van een model sepsis vroegtijdig voorspellen, zodat zorgverleners sneller kunnen ingrijpen, de zorgkwaliteit verbetert en de werkdruk op verpleegkundigen en artsen afneemt?*
# 
# Op basis van het onderzoek en de ontwikkelde modellen kan worden geconcludeerd dat een machine learning model in staat is om op basis van klinische patiëntgegevens voorspellingen te doen over de aanwezigheid van sepsis. Na zes iteraties is met het `all_lgbm_13-05-2026`-model een stabiel en betrouwbaar model ontwikkeld dat realistisch presteert.
# 
# ## Resultaten per cyclus
# | Cyclus | Model | Bevinding |
# |--------|-------|-----------|
# | I | Decision Tree / Random Forest | Schijnbaar perfecte scores door data leakage en onjuiste datasplitsing |
# | II | Decision Tree / Random Forest | Correcte patiëntgebaseerde splitsing; realistischere maar lagere scores |
# | III | ARIMA / SARIMA | Niet succesvol geïmplementeerd; tijdreeksmodellen niet geschikt voor multivariate input |
# | IV | Gradient Boosting / XGBoost | Verbeterde prestaties; vermoeden van resterende leakage via `Sepsis_Future` |
# | V | RF / GB / XGBoost + feature engineering | Grotere feature-set; onduidelijk welke configuratie bijdraagt |
# | VI | LightGBM (geoptimaliseerd) | Utility score ≈ 0.138; meest betrouwbaar en vrij van leakage |
# 
# ## Het beste model: `all_lgbm_13-05-2026`
# Het `all_lgbm_13-05-2026`-model is geselecteerd als het beste model op basis van de volgende argumenten:
# 
# **1. Hoge F1-score met balans tussen precision en recall**  
# Het model behaalt de hoogste F1-score van alle geteste modellen in Ronde 3, terwijl het tegelijk een relatief hoge recall behoudt. Dit is cruciaal in een klinische context: zoveel mogelijk echte sepsisgevallen worden herkend, zonder dat het aantal foutieve alarmen onbeheersbaar wordt. Dit sluit direct aan op de doelstelling uit de probleemanalyse om *alarmmoeheid* te minimaliseren.
# 
# **2. Geen data leakage**  
# In tegenstelling tot eerdere cycli wordt in Ronde 3 expliciet gebruik gemaakt van `SepsisLabel` als target in plaats van `Sepsis_Future`. Hiermee wordt de onbedoelde voorwaartse informatielekage voorkomen, wat leidt tot een realistischere utility score van ≈ 0.138.
# 
# **3. Uitlegbaarheid via feature importance**  
# LightGBM biedt via de `gain`-gebaseerde feature importance inzicht in welke vitale parameters en labwaarden het meest bijdragen aan de sepsisvoorspelling. Dit is essentieel voor transparantie richting zorgverleners, zoals beschreven in de ethische overwegingen.
# 
# **4. Efficiëntie**  
# LightGBM is aanzienlijk sneller te trainen dan CatBoost en biedt vergelijkbare of betere resultaten dan XGBoost in deze configuratie. Dit maakt het model ook praktisch inzetbaar.
# 
# ## Terugkoppeling naar het probleem
# Sepsis vereist vroege herkenning: ieder uur vertraging verhoogt de mortaliteit met 4–8%. Het ontwikkelde model kan zorgverleners ondersteunen door op basis van routinematig gemeten vitale waarden en laboratoriumuitslagen een risicomelding te genereren. Hiermee draagt het direct bij aan:
# 
# - **Sneller ingrijpen**: het model signaleert patronen die de klinische classificatie voorafgaan
# - **Minder gemiste gevallen**: de hoge recall beperkt het aantal onopgemerkte sepsispatiënten
# - **Lagere werkdruk**: automatische signalering vermindert de cognitieve belasting van verpleegkundigen
# - **Veiliger zorg**: consistente, datagedreven ondersteuning naast klinisch oordeel
# 
# ## Beperkingen en aanbevelingen
# Ondanks de positieve resultaten zijn er nog een aantal beperkingen:
# 
# - **Utility score**: Een score van ≈ 0.138 is nog relatief laag en biedt ruimte voor verbetering door verdere feature engineering en hyperparameteroptimalisatie.
# - **Ontbrekende variabelen**: De GCS-score ontbreekt in de dataset, waardoor de qSOFA- en SOFA-berekeningen minder nauwkeurig zijn dan de klinische standaard.
# - **Generaliseerbaarheid**: Het model is getraind op één dataset. Validatie op een externe dataset is noodzakelijk voordat het in de klinische praktijk kan worden ingezet.
# - **Bias-analyse**: Er is nog geen formele bias-analyse uitgevoerd op patiëntkenmerken zoals leeftijd en geslacht, wat een vereiste is conform de ethische uitgangspunten.
# 
# **Aanbeveling**: In een volgende iteratie wordt aangeraden om het model te valideren op externe data, een SHAP-analyse uit te voeren voor individuele voorspellings-uitleg, en samen te werken met klinische specialisten van Isala om de drempelwaarde af te stemmen op de praktijk.


