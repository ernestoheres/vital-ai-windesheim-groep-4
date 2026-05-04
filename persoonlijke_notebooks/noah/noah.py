# %% [markdown]
# # Analyse
# ## Vraagstelling
# 
# ## Het probleem
# 
# ## Doel van het onderzoek
# 
# ## Hypothese
# 
# # Ethiek
# 
# # Cycle I
# 
# ## Voorbereiding
# Voor deze stap heb ik alvast een aantal voorbereidingen getroffen om het werken met de dataset te vereenvoudigen. Het bepalen van sepsis gebeurt op basis van twee scores: de qSOFA-score en de SOFA-score. In de dataset (`test_data.csv`) ontbreken echter enkele variabelen die nodig zijn om deze scores volledig volgens de standaarddefinities te berekenen. Hieronder volgt een overzicht van de beperkingen:
# 
# qSOFA-score
# De qSOFA-score heeft normaal een bereik van 0–3, maar in deze dataset is dit beperkt tot 0–2, omdat:
# - De mentale status (Glasgow Coma Scale, GCS) ontbreekt.
# 
# SOFA-score
# Ook voor de SOFA-score zijn niet alle componenten volledig beschikbaar:
# - Het centrale zenuwstelsel (CNS) kan niet worden meegenomen, aangezien de GCS ontbreekt.
# - Voor respiratie wordt de verhouding SpO₂/FiO₂ gebruikt in plaats van de gebruikelijke PaO₂/FiO₂. Deze benadering is minder nauwkeurig.
# - De cardiovasculaire (CV) score is slechts gedeeltelijk beschikbaar. Alleen de Mean Arterial Pressure (MAP) wordt gebruikt, waardoor deze component maximaal 1 punt kan bijdragen, terwijl dit normaal kan oplopen tot 4 punten op basis van meerdere metingen.
# 
# Om consistent met deze beperkingen om te gaan, is de klasse `SofaCalculator` ontwikkeld. Deze klasse bevat alle logica voor het berekenen van de qSOFA- en SOFA-scores op basis van de beschikbare data. Door deze klasse te importeren in de notebooks, kunnen alle groepsleden dezelfde berekeningsmethode hanteren en wordt inconsistentie in de analyses voorkomen.

# %%
import sys
import os

sys.path.append(os.path.abspath("../../src"))

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../../data/test_data.csv', sep=',')

# %% [markdown]
# De dataset bevat een groot aantal variabelen, namelijk 43 kolommen. Het merendeel hiervan bestaat uit medische metingen die bij patiënten zijn uitgevoerd.
# 
# Daarnaast zijn er twee kolommen die betrekking hebben op tijd:
# - `HospAdmTime`: HIER UITLEG
# - `hour`: De duratie van het ziekenhuisbezoek.
# 
# Verder bevat de dataset één kolom zonder naam, waardoor onduidelijk is welke variabele deze vertegenwoordigt en hoe de bijbehorende waarden geïnterpreteerd moeten worden. Ook de kolom `Gender` is niet eenduidig te interpreteren, aangezien de codering van de waarden niet is gespecificeerd.
# 
# Op basis van een eerste verkenning van de data (bijvoorbeeld via een `describe`-overzicht) lijken de meeste waarden binnen een realistisch bereik te vallen. Hoewel ik geen medisch specialist ben, lijken eventuele uitschieters mogelijk binnen een ziekenhuiscontext, waar extreme waarden in sommige gevallen kunnen voorkomen.
# 
# Wat wel opvalt, is dat de `hour`-kolom waarden bevat tot maximaal 335. Het is niet direct duidelijk hoe deze geïnterpreteerd moeten worden (bijvoorbeeld als uren sinds opname), waardoor het lastig is om te beoordelen of deze waarden correct zijn. Dit brengt enige onzekerheid met zich mee over de interpretatie en betrouwbaarheid van deze specifieke tijdsvariabele.
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
# Hieronder staat een kleine helperfunctie waarmee het percentage aanwezige waarden in de dataset wordt berekend.
# 
# In de eerdere visualisatie was al zichtbaar dat waarde 1 in de `Gender`-kolom vaker voorkomt dan waarde 0. Uit nieuwsgierigheid is vervolgens precies berekend hoe groot dit verschil is in procenten.

# %%
def getGenderPercentage(data: pd.DataFrame) -> None: 
    genderLenght = len(data['Gender'])
    gender_vals = data['Gender'].unique()

    def calcPercentage(val: int) -> float:
        return (val /genderLenght) * 100

    for val in gender_vals:
        count = (data['Gender'] == val).sum()
        print(f'{val}: {calcPercentage(count):.2f}%')

getGenderPercentage(df)

# %% [markdown]
# uit dit pariplot durf ik geen conclusies te trekken omdat deze gigantisch is. Een heatmap lijkt mij veel geschikter om de correlatie tussen verschillende gevens te bekijken.

# %%
# sns.pairplot(data=df.drop(columns=['Patient_ID', 'Hour', 'HospAdmTime', 'Unnamed: 0', 'Unit1', 'Unit2']))   

# %% [markdown]
# De heatmap hieronder is overzichtelijker dan de pairplot die daarna volgt. Er is gekozen voor een driehoekige weergave, zodat dubbele waarden niet worden getoond en de visualisatie compacter blijft.
# 
# Annotaties zijn weggelaten, omdat deze de grafiek opnieuw onoverzichtelijk zouden maken, vergelijkbaar met de pairplot.

# %%
# corr = data=df.drop(columns=['Patient_ID', 'Hour', 'HospAdmTime', 'Unnamed: 0', 'Unit1', 'Unit2']).corr(numeric_only=True)
# mask = np.triu(np.ones_like(corr, dtype=bool))

# plt.figure(figsize=(14, 12))

# sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm')

# %% [markdown]
# Zoals te zien is, bevat de dataset meerdere observaties per Patient_ID, wat erop wijst dat per patiënt meerdere metingen over de tijd zijn vastgelegd. Daarnaast is het mogelijk dat één patiënt meerdere ziekenhuisopnames in de dataset heeft.

# %%
df.duplicated(subset=['Patient_ID']).sum()

# %%
test = df[['Patient_ID', 'HospAdmTime', 'Hour']].sort_values(by=['Patient_ID', 'Hour'])
test.head(35)

# %%
(test['Patient_ID'] == 2).sum()

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
# De enige aanvullende kolommen die nodig zijn, worden berekend en toegevoegd aan de dataset door de `SofaCalculator`. De twee functieaanroepen van de `SofaCalculator` gebruiken vervolgens de gedeeltelijke SOFA-scores om te bepalen of een patiënt wel of geen sepsis heeft.

# %%
from scepsis_prediction.SofaCalculator import SofaCalculator

sofa_calc = SofaCalculator(df)

df = sofa_calc.calculate_all_values()
df['SepsisLabel'] = sofa_calc.hasSepsis()

df.info()

# %% [markdown]
# Hieronder wordt berekend welk percentage van de dataset sepsis heeft en welk percentage niet. Daarbij wordt geen rekening gehouden met het feit dat wanneer bij een patiënt eenmaal sepsis is vastgesteld, deze in de daaropvolgende uren ook sepsis heeft. De berekening gebeurt dus per afzonderlijke rij in de dataset.
# 
# Hiervoor is een helpermethode ontwikkeld, die ook op andere momenten in het notebook kan worden aangeroepen om de verdeling opnieuw te controleren na eventuele aanpassingen.

# %%
def get_sepsis_values(data: pd.DataFrame) -> None:
    if 'SepsisLabel' not in data.columns:
        raise ValueError("Kolom 'SepsisLabel' ontbreekt in de DataFrame")

    counts = data['SepsisLabel'].value_counts().sort_index()
    total = len(data)

    print('Sepsis verdeling:')
    for val, count in counts.items():
        percentage = (count / total) * 100
        print(f'{val}: aantal = {count}, percentage = {percentage:.2f}%')

get_sepsis_values(df)

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

x_train, x_test, y_train, y_test = train_test_split(
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
model.fit(x_train, y_train)

# %% [markdown]
# De prestaties van het model zijn uitzonderlijk goed, maar vereisen nadere controle om te bevestigen of dit correct is.

# %%
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# De visualisatie laat zien hoeveel patiënten in de trainingsdataset op basis van de SOFA-scores als sepsis worden geclassificeerd. Daarnaast zijn er enkele andere variabelen opgenomen, maar deze lijken weinig tot geen invloed te hebben op deze classificatie.

# %%
f_model = dtreeviz.model(
    model,
    x_train,
    y_train,
    feature_names=x_train.columns,
    target_name='Sepsis',
    class_names=["No Sepsis", "Sepsis"]
)

f_model.view(scale=2)

# %% [markdown]
# ## Random forest
# Net als bij het vorige model wordt ook dit model getraind om het sepsislabel te herkennen. Dit gebeurt op een vergelijkbare manier als bij de decision tree.

# %%
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=10,           
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1                
)

rf_model.fit(x_train, y_train)

# %% [markdown]
# Net zoals bij het voirge lijkt dit model perfect. Dit is heel onwaarschijnlijk en moet dus ook nader na gekekn worden of dit wel correct is.

# %%
y_pred = rf_model.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# Omdat dit model, net als een decision tree, gebruikmaakt van boomstructuren, kijkt het niet naar één enkele boom maar naar meerdere bomen om te bepalen of iemand wel of niet aan het sepsislabel voldoet. Door deze combinatie van meerdere bomen kan het model verschillende patronen meenemen in de beslissing. Daarom lijkt dit model geschikter dan een enkele decision tree, omdat het op basis van meer informatie en perspectieven tot een eindbeslissing komt.

# %%
rf_tree = rf_model.estimators_[0]

rf_tree_model = dtreeviz.model(
    rf_tree,
    x_train,
    y_train,
    feature_names=x_train.columns,
    target_name='SepsisLabel',
    class_names=["No Sepsis", "Sepsis"]
)

rf_tree_model.view()

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
df = df.drop(columns=['qSOFA_partial', 'SOFA_modified_total', 'SepsisLabel'])

# %% [markdown]
# Hieronder worden de herberekeningen van deze waarden weergegeven. Dit keer worden de extra kolommen die nodig zijn voor de berekeningen wel in de dataset opgenomen. Daarnaast wordt het `sepsislabel` opnieuw toegevoegd aan de dataset.

# %%
sofa_calc = SofaCalculator(df)

df = sofa_calc.calculate_all_values(True, True)
df['SepsisLabel'] = sofa_calc.hasSepsis()

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
# In de vorige cyclus waren de modellen relatief eenvoudig opgezet. Daarbij werd direct gekeken naar de `SOFA`- en `qSOFA`-scores om te bepalen of een patiënt sepsis had. Omdat deze scores zelf al gebaseerd zijn op klinische criteria voor sepsis, leidde dit tot ogenschijnlijk perfecte (100%) voorspellingen.
# 
# In deze cyclus worden deze samengestelde scores bewust weggelaten. In plaats daarvan wordt uitsluitend gekeken naar de onderliggende medische parameters die samen de SOFA- en qSOFA-scores vormen. Op basis van deze ruwe gegevens wordt vervolgens bepaald of een patiënt sepsis heeft, met als doel een robuuster en realistischer voorspellingsmodel te ontwikkelen. Daarnaast ligt de focus ditmaal niet alleen op het vaststellen of een patiënt op dit moment sepsis heeft, maar juist op het voorspellen of een patiënt dit in de toekomst zal ontwikkelen.

# %%
qsofa_features = [col for col in df.columns if 'qsofa' in col.lower()]

sofa_features = [
    col for col in df.columns
    if 'sofa' in col.lower() and col not in qsofa_features
] + ['SF_ratio']

unit_features = [col for col in df.columns if 'unit' in col.lower()]

drop_cols = [
    'Patient_ID',
    'SepsisLabel', 
    'Sepsis_Future',
    *qsofa_features, 
    *sofa_features, 
    *unit_features
]

# %% [markdown]
# In de vorige cyclus werd het opsplitsen van de data niet correct uitgevoerd. De dataset werd willekeurig verdeeld, waardoor dezelfde patiënten zowel in de `train`- als in de `test`-set terechtkwamen, met hun verschillende bezoeken verspreid over beide sets. Dit is onlogisch en leidt bovendien tot een vertekende en onevenwichtige verdeling van de data.
# 
# In deze cyclus wordt dit probleem aangepakt door te splitsen op basis van unieke `Patient_ID`s. Hierdoor blijven alle gegevens van een individuele patiënt binnen één dataset (train of test), wat zorgt voor een realistischer en betrouwbaarder evaluatie van het model.

# %%
patients = df['Patient_ID'].unique()

train_patients, test_patients = train_test_split(
    patients, test_size=0.2, random_state=42
)

# %% [markdown]
# Hieronder worden de uiteindelijke `train`- en `test`-sets samengesteld. De gegevens worden hierbij gefilterd op basis van de unieke `Patient_ID`s, zodat elke dataset uitsluitend uit verschillende patiënten bestaat en er geen overlap tussen beide sets optreedt.
# 
# Daarnaast worden in deze stap ook de `X` en `y` gescheiden voor zowel de train- als testset.

# %%
train_df = df[df['Patient_ID'].isin(train_patients)]
test_df = df[df['Patient_ID'].isin(test_patients)]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df['Sepsis_Future'].astype(int)

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['Sepsis_Future'].astype(int)

# %% [markdown]
# ## Modeling
# 
# ### Decision Tree

# %%
print(X.shape)
print(y.shape)

# %%
model = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

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
# ### Random Forest

# %%
rf2_model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=10,           
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1                
)

rf2_model.fit(X_train, y_train)

# %%
y_pred = rf2_model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %%
rf2_tree = rf2_model.estimators_[0]

rf2_tree_model = dtreeviz.model(
    rf2_tree,
    X_train,
    y_train,
    feature_names=X_train.columns,
    target_name='SepsisLabel',
    class_names=["No Sepsis", "Sepsis"]
)

rf2_tree_model.view()

# %% [markdown]
# ## Reflectie

# %% [markdown]
# # Cycle III

# %% [markdown]
# ## Voorbereiding

# %% [markdown]
# ## Data preperation
# ### Clean Data

# %% [markdown]
# ### Construct Data

# %% [markdown]
# ### Format Data

# %% [markdown]
# ## Modeling

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 


