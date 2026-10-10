PawanSthal — Team Task Split
Oct 10, 2026 · @Someone
Start the GWA download and turbine labelling now: everything else depends on them. None of the training needs a GPU; a laptop or Google Colab is enough.
Training work
Three modules need training; all can start in parallel today.
Module
What is trained
What it needs
Time
M1: Turbine class classifier
Small RandomForest (or simple rules)
40+ turbines labelled old / mid / modern from satellite view (old = small, tightly packed; modern = huge, far apart). Training takes seconds.
1–1.5 h of labelling
M2: Wind downscaling
LightGBM regressor
GWA rasters + NASA POWER data downloaded first. Training takes minutes.
1.5–2 h incl. data prep
M5: AEP surrogate (stretch)
Small neural net or gradient boosting
2,000–5,000 PyWake simulations as training data; needs M3 working first. Run on Colab or a spare laptop.
1–3 h of compute
Compute-heavy, not training: the M4 optimiser precompute. Each of the 6–10 scenarios can take 10–30 minutes, so give it to whoever has the fastest machine.
Best split for a teammate: M1 labelling plus M2 training; then M5 data generation on Colab once M3 works.
Work that needs a person, not code
Data (do first, it blocks everything)
[ ] Download the GWA GeoTIFFs by hand from globalwindatlas.info: India, wind speed at 100 m and 150 m, elevation, ruggedness. The site has no download API.
[ ] Confirm the Muppandal bounding box on Google Maps.
[ ] Hand-digitise turbines in geojson.io (satellite view) if OpenStreetMap has fewer than about 30 in the box. Can be done together with M1 labelling.
[ ] Look up and cite four numbers for config.py:
    ◦ CEA grid CO₂ emission factor (CEA CO₂ Baseline Database, latest version)
    ◦ Wind CAPEX in ₹ crore per MW (MNRE/CERC or a recent news source)
    ◦ A Tamil Nadu wind tariff (TNERC order) for payback
    ◦ Average household electricity use (kWh/year) for “homes powered”
[ ] Set defaults for owner count in the cluster and whether it is on 11 kV (feeds the priority score).
AWS and admin
[ ] AWS account setup (debit card or RuPay works; verification costs about ₹2).
[ ] AWS Builder Center student verification — required to compete, every team member, today.
[ ] Request the $25 credits if needed (team leader, one form per team).
[ ] Check the exact Sunday submission deadline on the schedule page and plan backwards from it.
Presentation (Sunday)
[ ] Video: script, voiceover and screen recording, about 3 minutes.
[ ] Architecture image and README.
[ ] Builder Center blog post (optional; top 5 posts win AirPods).
Team split (4 people)