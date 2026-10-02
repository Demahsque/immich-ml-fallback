# Protocole ML d'Immich utilisé par le proxy

Relevé dans le code source d'Immich **v3.2.4** (commit `db355f7`). Si une future version d'Immich change ce format, c'est ici qu'il faut regarder en premier.

## Côté serveur Immich (client)

`server/src/repositories/machine-learning.repository.ts`

- Recherche texte : `encodeText(text, { language, modelName })` envoie
  `POST {url}/predict`, corps **multipart/form-data** (`FormData` de Node) :
  - `entries` = `{"clip":{"textual":{"modelName":"<modèle>","options":{}}}}` (`language: undefined` disparaît du JSON) ;
  - `text` = la requête de l'utilisateur, telle quelle.
- Réponse attendue : `{"clip": "<chaîne>"}` ; la chaîne est transmise telle quelle à Postgres comme vecteur pgvector (`smart_search.embedding <=> …`, distance cosinus, `vector_cosine_ops`).
- Sélection du serveur (`predict`) : `urls.filter(sain)` puis `urls.filter(non sain)` ; chaque `fetch` n'a **pas** de timeout explicite ; toute réponse non-OK ou exception → `setHealthy(url, false)` et passage à l'URL suivante ; si toutes échouent, la requête échoue.
- Santé : toutes les `availabilityChecks.interval` ms (défaut 30 000), `GET {url}/ping` avec timeout `availabilityChecks.timeout` (défaut 2 000) ; OK = `response.ok`. Avec les vérifications désactivées, toutes les URLs sont considérées saines (ordre de la liste).
- Cache : `search.service.ts` garde les 100 derniers embeddings (clé : modèle + requête + langue).
- Autres tâches envoyées au même `/predict` : `clip/visual` (image), `facial-recognition/{detection,recognition}`, `ocr/{detection,recognition}`.

## Côté serveur ML d'Immich (ce que le proxy imite)

`machine-learning/immich_ml/main.py`, `schemas.py`, `models/clip/textual.py`, `models/transforms.py`

- `GET /` → `{"message":"Immich ML"}` ; `GET /ping` → texte `pong`.
- `POST /predict` : `entries` (Form, JSON, 422 `Invalid request format.` si invalide), `image` (fichier) ou `text` (Form) ; ni l'un ni l'autre → 400.
- Réponse : `{<task>: <sortie>}` ; pour le texte CLIP, `<task>` = `clip` et la sortie est `orjson.dumps(vecteur float32, OPT_SERIALIZE_NUMPY).decode()` : un **tableau JSON sérialisé dans une chaîne**.
- Pré-traitement texte CLIP : espaces normalisés ; pour les modèles « canonicalize » (SigLIP) : ponctuation retirée et minuscules. **Aucun gabarit de prompt** (`a photo of …`) n'est ajouté : la requête brute est encodée.
- Nom du modèle : `clean_name` = dernier segment après `/`, avec `:`, `\`, `/` → `_` et `.` supprimés ; le proxy applique la même normalisation avant de comparer.

## Ce que fait le proxy

| Requête | Réponse |
|---|---|
| `GET /ping` | `pong` (200), ou 503 sans vecteurs chargés |
| `POST /predict`, `clip/textual`, bon modèle, au moins un mot connu | 200 `{"clip":"[…]"}` |
| même chose, mauvais modèle | 409 |
| même chose, aucun mot connu | 422 |
| `clip/visual`, visages, OCR, ou plusieurs tâches | 501 |
| image jointe, ou ni `text` ni `image` | 400 |
| `entries` absent / JSON invalide | 422 |
