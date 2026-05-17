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
from helpers.notebook_helpers import read_dataset

df = read_dataset()

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
# plt.figure(figsize=(14, 12))

# sns.boxplot(data=df.drop(columns=['Patient_ID', 'Hour', 'HospAdmTime', 'Unnamed: 0', 'Unit1', 'Unit2']))
# plt.ylim(-20, 450)
# plt.xticks(rotation=90)

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
y_test_data = df[['Patient_ID', 'HospAdmTime', 'Hour']].sort_values(by=['Patient_ID', 'Hour'])
y_test_data.head(35)

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
# De enige aanvullende kolommen die nodig zijn, worden berekend en toegevoegd aan de dataset door de `SofaCalculator`. De twee functieaanroepen van de `SofaCalculator` gebruiken vervolgens de gedeeltelijke SOFA-scores om te bepalen of een patiënt wel of geen sepsis heeft.

# %%
# from scepsis_prediction.SofaCalculator import SofaCalculator

# sofa_calc = SofaCalculator(df)

# df = sofa_calc.calculate_all_values()
# #df['SepsisLabel'] = sofa_calc.hasSepsis()

# df.info()

# %% [markdown]
# Hieronder wordt berekend welk percentage van de dataset sepsis heeft en welk percentage niet. Daarbij wordt geen rekening gehouden met het feit dat wanneer bij een patiënt eenmaal sepsis is vastgesteld, deze in de daaropvolgende uren ook sepsis heeft. De berekening gebeurt dus per afzonderlijke rij in de dataset.
# 
# Hiervoor is een helpermethode ontwikkeld, die ook op andere momenten in het notebook kan worden aangeroepen om de verdeling opnieuw te controleren na eventuele aanpassingen.

# %%
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
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import dtreeviz

# model = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
# model.fit(X_train, y_train)

# %% [markdown]
# De prestaties van het model zijn uitzonderlijk goed, maar vereisen nadere controle om te bevestigen of dit correct is.

# %%
# y_pred = model.predict(X_test)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# De visualisatie laat zien hoeveel patiënten in de trainingsdataset op basis van de SOFA-scores als sepsis worden geclassificeerd. Daarnaast zijn er enkele andere variabelen opgenomen, maar deze lijken weinig tot geen invloed te hebben op deze classificatie.

# %%
# dt1_model = dtreeviz.model(
#     model,
#     X_train,
#     y_train,
#     feature_names=X_train.columns,
#     target_name='Sepsis',
#     class_names=["No Sepsis", "Sepsis"]
# )

# dt1_model.view(scale=2)

# %% [markdown]
# ## Random forest
# Net als bij het vorige model wordt ook dit model getraind om het sepsislabel te herkennen. Dit gebeurt op een vergelijkbare manier als bij de decision tree.

# %%
# from sklearn.ensemble import RandomForestClassifier

# rf1_model = RandomForestClassifier(
#     n_estimators=100,        
#     max_depth=10,           
#     class_weight='balanced', 
#     random_state=42,
#     n_jobs=-1                
# )

# rf1_model.fit(X_train, y_train)

# %% [markdown]
# Net zoals bij het voirge lijkt dit model perfect. Dit is heel onwaarschijnlijk en moet dus ook nader na gekekn worden of dit wel correct is.

# %%
# y_pred = rf1_model.predict(X_test)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# Omdat dit model, net als een decision tree, gebruikmaakt van boomstructuren, kijkt het niet naar één enkele boom maar naar meerdere bomen om te bepalen of iemand wel of niet aan het sepsislabel voldoet. Door deze combinatie van meerdere bomen kan het model verschillende patronen meenemen in de beslissing. Daarom lijkt dit model geschikter dan een enkele decision tree, omdat het op basis van meer informatie en perspectieven tot een eindbeslissing komt.

# %%
# rf1_tree = rf1_model.estimators_[0]

# rf1_tree_model = dtreeviz.model(
#     rf1_tree,
#     X_train,
#     y_train,
#     feature_names=X_train.columns,
#     target_name='SepsisLabel',
#     class_names=["No Sepsis", "Sepsis"]
# )

# rf1_tree_model.view()

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
# df = df.drop(columns=['qSOFA_partial', 'SOFA_modified_total'])

# %% [markdown]
# Hieronder worden de herberekeningen van deze waarden weergegeven. Dit keer worden de extra kolommen die nodig zijn voor de berekeningen wel in de dataset opgenomen. Daarnaast wordt het `sepsislabel` opnieuw toegevoegd aan de dataset.

# %%
# sofa_calc = SofaCalculator(df)

# df = sofa_calc.calculate_all_values(True, True)
# #df['SepsisLabel'] = sofa_calc.hasSepsis()

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
# TODO: Tekst hierboven bij werken

# %% [markdown]
# In de vorige cyclus werd het opsplitsen van de data niet correct uitgevoerd. De dataset werd willekeurig verdeeld, waardoor dezelfde patiënten zowel in de `train`- als in de `test`-set terechtkwamen, met hun verschillende bezoeken verspreid over beide sets. Dit is onlogisch en leidt bovendien tot een vertekende en onevenwichtige verdeling van de data.
# 
# In deze cyclus wordt dit probleem aangepakt door te splitsen op basis van unieke `Patient_ID`s. Hierdoor blijven alle gegevens van een individuele patiënt binnen één dataset (train of test), wat zorgt voor een realistischer en betrouwbaarder evaluatie van het model.
# 
# De helperfunctie `train_test_split_by_patient` is ontwikkeld om de dataset op een consistente en correcte manier te splitsen. Het doel van deze functie is om herbruikbaarheid te waarborgen, zodat bij toekomstige iteraties of aanpassingen aan de data dezelfde logica eenvoudig opnieuw kan worden toegepast. Hierdoor wordt voorkomen dat identieke bewerkingen telkens opnieuw moeten worden uitgevoerd en dat dezelfde code herhaaldelijk herschreven moet worden.

# %% [markdown]
# Hieronder word het splitsen van de data daadwerkelijk gedaan op basis van de helperfunctie.

# %%
train_patients, test_patients = train_test_split_by_patient(df)

# %% [markdown]
# Hieronder worden de uiteindelijke `train`- en `test`-sets samengesteld. De gegevens worden gefilterd op basis van unieke `Patient_ID`s, zodat beide datasets uitsluitend uit verschillende patiënten bestaan en er geen overlap optreedt. Dit proces wordt uitgevoerd met behulp van de helperfunctie `get_train_test_data_by_patient`, waardoor dezelfde bewerkingen eenvoudig herhaald kunnen worden door het hele notebook.
# 
# Daarnaast worden in deze stap de invoervariabelen (`X`) en de doelvariabele (`y`) gescheiden voor zowel de train- als de testset.

# %% [markdown]
# Hieronder worden de `train`- en `test`-sets opgehaald op basis van de huidige dataset. Voor deze voorspelling is `Patient_ID` overbodig en word deze daarom nog verwijderd.

# %%
X_train, y_train, train_patient_ids, X_test, y_test, test_patient_ids = get_train_test_data_by_patient(df, train_patients, test_patients, delete_patient_ids=True)

# %% [markdown]
# ## Modeling
# In dit hoofdstuk worden dezelfde modellen op identieke wijze opgebouwd als in de vorige cyclus. Het verschil is dat er ditmaal gebruik wordt gemaakt van een beter voorbereide dataset. In hoeverre dit daadwerkelijk tot betere prestaties leidt, zal blijken uit de modelresultaten, al is de verwachting dat de verbeterde datakwaliteit hier een positieve bijdrage aan levert.
# 
# ### Decision Tree
# Hieronder wordt de opbouw van de nieuwe decision tree weergegeven. Hierbij ligt de focus met name op de `classification report` en de `confusion matrix`, om te beoordelen of het model daadwerkelijk is verbeterd. Het model wordt getraind en geëvalueerd met de eerder samengestelde `train`- en `test`-datasets.

# %%
# model = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
# model.fit(X_train, y_train)

# %% [markdown]
# Uit de resultaten blijkt dat het model minder goed presteert dan in de vorige cyclus. Met name de lage precision voor klasse 1 en de relatief hoge hoeveelheid fout-positieve voorspellingen laten zien dat het model moeite heeft om sepsis nauwkeurig te identificeren. Tegelijkertijd is de recall voor klasse 1 wel hoog, wat betekent dat een groot deel van de daadwerkelijke gevallen wordt herkend, maar ten koste van veel onterechte voorspellingen.
# 
# Hoewel de prestaties op het eerste gezicht slechter lijken, is dit juist een positieve ontwikkeling. In de vorige cyclus was er sprake van datalekken en een onjuiste datasplitsing, waardoor de resultaten te optimistisch waren. In de huidige aanpak is de data correct verdeeld op basis van unieke patiënten, wat zorgt voor een realistischer en betrouwbaarder beeld van de modelprestaties.

# %%
# y_pred_dt2 = model.predict(X_test)

# print(classification_report(y_test, y_pred_dt2))
# print(confusion_matrix(y_test, y_pred_dt2))

# %% [markdown]
# Kijkend naar het model valt op dat het slechts een beperkt aantal kenmerken gebruikt om te bepalen of iemand sepsis heeft of niet. Hierdoor lijkt het model onvoldoende complex en daarmee minder geschikt voor deze vraagstelling. Tegelijkertijd is het wel positief om te zien dat, ondanks deze beperkingen, er sprake is van een verbetering ten opzichte van de vorige aanpak.

# %%
# dt2_model = dtreeviz.model(
#     model,
#     X_train,
#     y_train,
#     feature_names=X_train.columns,
#     target_name='Sepsis',
#     class_names=["No Sepsis", "Sepsis"]
# )

# dt2_model.view(scale=2)

# %%
# export_prediction_set(test_patient_ids, df, y_pred_dt2)
# print_utiltiy_score()

# %%
# utility = evaluate_sepsis_score(
#     "testset (with label).csv",
#     "predictions.csv"
# )

# print(utility)

# %% [markdown]
# ### Random Forest
# Hetzelfde geldt hier. Het model wordt op dezelfde manier opgebouwd als het vorige, maar ditmaal met gebruik van de nieuwe `train`- en `test`datasets.

# %%
# rf2_model = RandomForestClassifier(
#     n_estimators=100,        
#     max_depth=3,           
#     class_weight='balanced', 
#     random_state=42,
#     n_jobs=-1                
# )

# rf2_model.fit(X_train, y_train)

# %% [markdown]
# Uit de resultaten van het random forest blijkt dat het model beter presteert dan de decision tree. De accuracy ligt hoger en ook de verdeling in de confusion matrix laat zien dat er minder fout-positieve voorspellingen worden gedaan. Dit wijst erop dat het model beter in staat is om onderscheid te maken tussen de klassen.
# 
# Tegelijkertijd blijft de precision voor klasse 1 laag, wat betekent dat een groot deel van de voorspelde sepsisgevallen onterecht is. De recall voor klasse 1 is daarentegen nog steeds redelijk hoog, waardoor het model wel een groot deel van de daadwerkelijke gevallen weet te identificeren.
# 
# Net als bij het vorige model geldt dat de prestaties realistischer zijn dan in de eerdere cyclus. Door de verbeterde datasplitsing op basis van unieke patiënten geeft dit model een betrouwbaarder beeld van de werkelijke prestaties, ook al zijn deze minder optimaal dan voorheen.

# %%
# y_pred_rf2 = rf2_model.predict(X_test)

# print(classification_report(y_test, y_pred_rf2))
# print(confusion_matrix(y_test, y_pred_rf2))

# %% [markdown]
# Uit deze visualisatie blijkt dat het model rekening houdt met een veel meer variabelen uit de dataset. In plaats van slechts enkele kenmerken, worden meerdere factoren meegenomen die gezamenlijk bijdragen aan de classificatie of iemand wel of geen sepsis heeft.
# 
# Daarnaast valt op dat ook variabelen zoals `hour` worden meegenomen. Dit geeft waardevol inzicht in het tijdsaspect van de metingen en laat zien op welke momenten bepaalde waarden een rol spelen bij het voorspellen van sepsis. Hierdoor wordt er als het ware per uur gekeken naar de kans dat een patiënt de aandoening heeft of zal ontwikkelen.

# %%
# rf2_tree = rf2_model.estimators_[0]

# rf2_tree_model = dtreeviz.model(
#     rf2_tree,
#     X_train.drop(columns=['Patient_ID']),
#     y_train,
#     feature_names=X_train.columns,
#     target_name='SepsisLabel',
#     class_names=["No Sepsis", "Sepsis"]
# )

# rf2_tree_model.view(scale=2)

# %%
# export_prediction_set(test_patient_ids, df, y_pred_rf2)
# print_utiltiy_score()

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

# %% [markdown]
# ## Voorbereiding

# %% [markdown]
# ## Data preperation
# ### Clean Data
# Deze stap word in deze cycle overgeslagen omdat er verder word gewerkt met de data van de vorige cyclus. 

# %% [markdown]
# ### Construct Data
# Hieronder staat een kleine helpermethode die de duur van een bezoek berekent, oftewel het tijdsverschil (tijdsdeltа) van een opname. In de dataset begint de tijdsregistratie van een bezoek bij 0. Daarom wordt hier de maximale waarde genomen en met 1 verhoogd. Dit geeft de totale duur weer dat een patiënt in het ziekenhuis heeft verbleven. Door deze berekening toe te passen, wordt het eenvoudiger om de `ARIMA`- en `SARIMA`-model verder uit te werken.
# 
# Deze functie geeft een `Series` terug oftewel een kolom. Dit omdat binnen de functie gefilterd word op bepaalde data. Daarom is het makkelijker om deze terug tegeven inplaats van een geheel `DataFrame`.

# %% [markdown]
# Omdat in de functie `get_train_test_data_by_patient` de `Patient_ID` wordt verwijderd, moeten de basisvoorbereidingen opnieuw worden uitgevoerd. Dit is noodzakelijk omdat `calculateVisitTime` wordt berekend op basis van zowel de `Patient_ID` als de `Hour`. `Patient_ID` is namelijk in de vorige cycle uit de `train`- en `test`-data verwijderd.

# %%
train_patients, test_patients = train_test_split_by_patient(df)

X_train, y_train, _, X_test, y_test, _ = get_train_test_data_by_patient(df, train_patients, test_patients)

X_train['visit_duration'] = calculateVisitTime(X_train)
X_test['visit_duration'] = calculateVisitTime(X_test)

# %% [markdown]
# ### Format Data
# Deze stap word in deze cycle overgeslagen omdat er verder word gewerkt met de data van de vorige cyclus. 

# %% [markdown]
# ## Modeling
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
# Niet werkend gekregen met huidige situatie van mijn dataset. Deze cycle word hierbij afgerond. Word in de volgende gekeken naar een boost model.

# %% [markdown]
# # Cycle IV

# %% [markdown]
# ## Voorbereiding

# %%
X_train = X_train.drop(columns=['visit_duration'])
X_test = X_test.drop(columns=['visit_duration'])

# %% [markdown]
# ## Data preperation
# ### Clean Data

# %% [markdown]
# ### Construct Data

# %% [markdown]
# ### Format Data

# %% [markdown]
# ## Modeling
# 
# ### Gradient boosting

# %%
# from sklearn.ensemble import GradientBoostingClassifier

# gb1_model = GradientBoostingClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42
# )

# gb1_model.fit(X_train, y_train)

# %%
# y_pred_gb1 = gb1_model.predict(X_test)

# print(classification_report(y_test, y_pred_gb1))
# print(confusion_matrix(y_test, y_pred_gb1))

# %%
# export_prediction_set(test_patient_ids, df, y_pred_gb1)
# print_utiltiy_score()

# %% [markdown]
# ### Xgboost

# %%
# from xgboost import XGBClassifier

# xgb1_model = XGBClassifier(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric='logloss'
# )

# xgb1_model.fit(X_train, y_train)

# %%
# y_pred_xgb1 = xgb1_model.predict(X_test)

# print(classification_report(y_test, y_pred_xgb1))
# print(confusion_matrix(y_test, y_pred_xgb1))

# %%
# export_prediction_set(test_patient_ids, df, y_pred_xgb1)
# print_utiltiy_score()

# %% [markdown]
# ## Reflectie

# %% [markdown]
# Betere resultaten dan een Random Forest. Voor een betere toekomstig model moet er bv gekeken naar feature engineering en het verbeteren van de missende data. Omdat deze allemaal op 0 geplaast zijn.

# %% [markdown]
# # Cycle V
# In deze cycle wil ik de focus leggen op het opnieuw verwerken/resetten van de dataset. Dit voornamelijk om de missende waardes beter bij te vullen. Hierdoor hoop ik dat de medische waardes die ik bereken voor de `SOFA`-scores ook wat meer voorstellen dam dat ze nu doen. Grotendeels van de aanpassingen en toevoegingnen ik aan de dataset ga hier wel herhaald worden. Hiervoor ga ik in de vorige cycles daar wat functies voor aanmaken zodat ik die gewoon kan aanroepen en dat ik deze niet handmatig moet overneemen wat dubbele code verhelpt.
# 
# ## Voorbereiding
# Hier word eerst de data opnieuw ingelezen en wat andere simpele stappen die in voorgaande cycles ook is uitgevoerd.

# %%
# df = read_dataset()
# df = prep_dataset(df)

# %% [markdown]
# ## Data preperation
# ### Clean Data

# %% [markdown]
# ### Construct Data

# %%
from scepsis_prediction.feature_engineering import add_all_features

# df = add_all_features(df, include_rolling=True, include_temporal=True)

# df = df.ffill().bfill()

# %% [markdown]
# ### Format Data

# %%
# X_train, y_train, train_patient_ids, X_test, y_test, test_patient_ids = get_train_test_data_by_patient(df, train_patients, test_patients, delete_patient_ids=True)

# %% [markdown]
# ## Modeling
# 
# ### Random forest

# %%
# rf5_model = RandomForestClassifier(
#     n_estimators=100,        
#     max_depth=3,           
#     class_weight='balanced', 
#     random_state=42,
#     n_jobs=-1                
# )

# rf5_model.fit(X_train, y_train)

# %%
# y_pred_rf5 = rf5_model.predict(X_test)

# print(classification_report(y_test, y_pred_rf5))
# print(confusion_matrix(y_test, y_pred_rf5))

# %%
# rf2_tree = rf2_model.estimators_[0]

# rf2_tree_model = dtreeviz.model(
#     rf2_tree,
#     X_train.drop(columns=['Patient_ID']),
#     y_train,
#     feature_names=X_train.columns,
#     target_name='SepsisLabel',
#     class_names=["No Sepsis", "Sepsis"]
# )

# rf2_tree_model.view(scale=2)

# %%
# export_prediction_set(test_patient_ids, df, y_pred_rf5)
# print_utiltiy_score()

# %% [markdown]
# ### Gradient boosting

# %%
# gb5_model = GradientBoostingClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42
# )

# gb5_model.fit(X_train, y_train)

# %%
# y_pred_gb5 = gb5_model.predict(X_test)

# print(classification_report(y_test, y_pred_gb5))
# print(confusion_matrix(y_test, y_pred_gb5))

# %%
# export_prediction_set(test_patient_ids, df, y_pred_gb5)
# print_utiltiy_score()

# %% [markdown]
# ### Xgboost

# %%
# xgb5_model = XGBClassifier(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric='logloss'
# )

# xgb5_model.fit(X_train, y_train)

# %%
# y_pred_xgb5 = xgb5_model.predict(X_test)

# print(classification_report(y_test, y_pred_xgb5))
# print(confusion_matrix(y_test, y_pred_xgb5))

# %%
# export_prediction_set(test_patient_ids, df, y_pred_xgb5)
# print_utiltiy_score()

# %% [markdown]
# ## Reflectie
# In alle gevallen is de random forest slecht
# 
# # Cycle VI

# %% [markdown]
# 

# %%
import sqlite3

conn = sqlite3.connect("optuna_storage/sepsis_optuna.db")

# Bekijk eerst welke tabellen erin zitten
tables = pd.read_sql(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table';
    """,
    conn,
)

print(tables)

# %%
trials_df = pd.read_sql(
    "SELECT * FROM studies",
    conn,
)

print(trials_df.head())

# %%
values_df = pd.read_sql(
    "SELECT * FROM trial_values",
    conn,
)

print(values_df.head())

# %%
results_df = pd.read_csv('optuna_storage/results_1.csv', sep=',')

# %%
results_df.sort_values(by=['model', 'f1'])
results_df.head(15)

# %% [markdown]
# Features die worden verwijderd:
# Deze hebben overaL de slechte perfrormances als je naar de beste runs kijkt.
# - Base
# - temporal_only
# 
# Model dat word verwijder:
# Over het algemeen de slechte perferformance van allemaal
# - XGB

# %% [markdown]
# 

# %%
import joblib

from scepsis_prediction.feature_engineering import add_all_features
from helpers.notebook_helpers import (
    read_dataset, 
    prep_dataset, 
    get_train_test_data_by_patient, 
    train_test_split_by_patient, 
    get_train_test_data_by_patient,
    export_prediction_set,
    print_utiltiy_score
)

# %%
df = read_dataset()
df = prep_dataset(df, add_sepsis_future=True)
df = add_all_features(df=df, include_temporal=False, include_rolling=True)

df = df.ffill()
df = df.dropna()

train_patients, test_patients = train_test_split_by_patient(df)
_, _, _, X_test, _, test_patient_ids = get_train_test_data_by_patient(
    df,
    train_patients,
    test_patients,
    delete_patient_ids=True,
)

# %% [markdown]
# VERSIE 1 met sepsis_future

# %%
loaded = joblib.load('optuna_storage/saved_models/rolling_only_xgb_10-05-2026.pkl')

model = loaded["model"]
threshold = loaded.get("threshold", 0.5)

proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

# %%
export_prediction_set(test_patient_ids, df, y_pred)
print_utiltiy_score()

# %% [markdown]
# VERSIE 2 met sepsislabel

# %%
from helpers.notebook_helpers import run_model

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

# %%
from lightgbm import LGBMClassifier

loaded = joblib.load('optuna_storage/saved_models/all_lgbm_13-05-2026.pkl')

model = loaded["model"]

# %%
trained_features = model.feature_name_
current_features = X_test.columns.tolist()

missing_in_test = set(trained_features) - set(current_features)
extra_in_test = set(current_features) - set(trained_features)

print("Ontbreekt in X_test:")
print(missing_in_test)

print("\nExtra in X_test:")
print(extra_in_test)

# %%
threshold = loaded.get("threshold", 0.5)

proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

# %%
export_prediction_set(test_patient_ids, df, y_pred)
print_utiltiy_score()


