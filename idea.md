Track: Waste and Energy. It covers powering cities more cleanly. Remember how it’s judged: a small problem solved well beats a big one solved vaguely, and there is no live demo, so the video is what the judges see.

Recommended: “PawanSthal”, a wind repowering and layout optimiser for state nodal agencies

The government problem (citable):

India’s old wind farms sit on the best wind sites but run small, outdated turbines. NIWE estimates the country’s repowering potential at 25.406 GW, counting turbines below 2 MW. 
mondaq
Under the 2023 policy, State and Central Nodal Agencies may identify turbines that are candidates for repowering. So there is a named government user with a named task. 
mondaq
Progress is stuck. Repowering of old sub-2 MW turbines at the best sites has been slow, even after MNRE revised the policy in December 2023. Most old projects connect to 11 kV lines, and many wind farms have multiple turbine owners. 
anantamias
jmkresearch
Bad spacing costs real energy. NREL says wind farms can lose up to 20% of their gross energy production to wakes from upstream turbines. 
windpowerengineering
The national target makes this urgent. The government wants wind capacity above 100 GW by 2030, up from 58.52 GW as of August 31, 2026. 
energetica-india
energetica-india

What it does: An official draws or selects an old wind cluster, for example in Tamil Nadu or Gujarat. The app then:

Shows a heatmap of wind resource at hub height.
Scores each old turbine for repowering priority.
Proposes a new layout: fewer, larger turbines at optimised spacing.
Reports the energy and money gained.

This is your “optimised distance” idea, done properly.

Where the ML is (and the honest framing judges will respect):

Wind resource downscaling. Train a gradient-boosted model (LightGBM) or a small CNN. Inputs are coarse reanalysis wind (ERA5 or NASA POWER) plus terrain features (elevation, slope, roughness from an SRTM DEM). The target is high-resolution wind speed at 120–150 m. Use Global Wind Atlas rasters as the labels. Output: a fine-grained wind map anywhere in India.
Neural surrogate wake model. Physics libraries like PyWake (from DTU Wind Energy, Denmark) or NREL’s FLORIS compute wake losses accurately but slowly. Generate a few thousand layouts with them, then train a neural net that predicts annual energy from a layout in milliseconds. That speed is what makes live what-if sliders possible. It is also a real research technique, not ML for decoration.
Optimiser. Run a genetic algorithm or Bayesian optimisation over turbine positions, using the surrogate. Constraints: spacing (the industry rule of thumb is roughly 5–9 rotor diameters downwind and 3–5 crosswind), land boundary, setbacks from houses and roads, and cable distance to the nearest substation.
Repowering priority score. Rank old turbines by expected energy gain divided by cost.

Visuals and what-if scenarios:

Wind-speed heatmap and wake-deficit heatmap, shown as before and after.
Sliders for turbine model (e.g. 1.5 MW to 3–4 MW class), hub height (120 m vs 150 m), spacing in rotor diameters, and number of turbines.
Cards showing annual energy gained, CO₂ avoided, homes powered, and payback period.

Why it scales: The same pipeline works for any polygon in India. Gross potential is 1,163.9 GW at 150 m, concentrated in eight states led by Rajasthan, Gujarat, Maharashtra and Karnataka. The same tool can also screen new sites as well as repowering candidates. 
pib

This is the same skeleton as your solar site-viability idea: resource × land × distance-to-infrastructure × cost. If you already started that, much of it carries over.

Backup 1: “PawanCast”, a day-ahead wind forecasting and penalty simulator
Problem: All renewable projects must forecast and schedule output in 15-minute blocks a day ahead. CERC is tightening the rules: from April 2026 the tolerance margin shrinks each year until 2031, when renewable generators are treated like conventional plants. The wind producers’ association estimated some projects could lose up to 48% of their revenue. Developers also say IMD data updates only every six hours, which makes short-term forecasting very hard. 
bridgetoindia.com +3
ML: quantile forecasting (LightGBM or a Temporal Fusion Transformer) on public turbine SCADA datasets.
What-if: “If the tolerance band drops from X% to Y%, here is the penalty with a naive forecast versus our model.”
Strong ML, but it’s harder to make a visual, government-facing demo for it.
Backup 2: a curtailment predictor for state load dispatch centres
Problem: Tamil Nadu’s history. Peak-season curtailment averaged about 30–35% in 2012–15 and about 20–25% more recently. 
downtoearth
Weakness: public curtailment data is scarce, which is risky with two days left.
Two-day build plan for PawanSthal
When	Do
Fri night	Pick one real cluster, e.g. Muppandal (TN) or Kutch (GJ). Pull Global Wind Atlas tiles, NASA POWER data, an SRTM DEM, and OSM substations, roads and houses.
Sat AM	Train the downscaling model and render the heatmap. Run PyWake to generate layout training data.
Sat PM	Train the surrogate and build the GA optimiser. Compare old layout vs optimised layout.
Sun AM	Frontend: Leaflet map, sliders, result cards. Deploy on AWS.
Sun PM	Record the 3-minute video. Optionally write a Builder Center blog post, since the top 5 blogs win AirPods.

AWS stack (prizes need it): to be eligible for prizes, your project must use at least one AWS open-source tool or be deployed on AWS.

S3 for rasters.
Lambda or Fargate for the optimiser API.
Amplify for the frontend.
Optional: a Strands agent so an official can type “what if we use 150 m hubs and cap at 10 turbines?”. You already know Strands from Paani Radar.

Video hook: “India has 25 GW of wind on its best land running on 1990s turbines. Here’s what Muppandal produces if repowered: X GWh more, with no new land.”

Keep the scope to one cluster done end-to-end; that beats an all-India map that half works. If you’d like, I can scaffold the PyWake data generation and surrogate training code next.

Sources:

WeMakeDevs Environmental Hacks
MNRE repowering policy (via Energetica India)
Mondaq: Repowering Policy 2023
JMK Research: Repowering challenges
Anantam IAS: Repowering gap
PIB: India’s path to 100 GW wind
Energetica India: 8 GW target FY27
Wind Power Engineering: NREL wake losses
Reuters via WHTC: DSM rules
Reuters via MarketScreener: weather data gap
Bridge to India: DSM bands
Down To Earth: Curtailment in Tamil Nadu

is data for pawan sathal available and is it a real problem which judges feel worth solvinf

Searched the web

Short answer: yes, the data is free and good enough for a demo. Yes, it’s a real government problem. There is one data gap and one judging risk you should know about before you commit.

Is the data available?
What you need	Source	Status
Wind speed at 100/150 m	Global Wind Atlas (DTU + World Bank)	✅ Free. Raster files at 250 m resolution, CC-BY 4.0 licence, with layers for wind speed, power density, elevation, air density and ruggedness.
Wind direction time series (needed for wake modelling)	NASA POWER API (WS50M, WD50M, hourly)	✅ Free, no API key. ERA5 has 100 m wind but needs a CDS account and can queue for hours, so skip it today.
Terrain	SRTM DEM, or the elevation band in GWA	✅ Free
Roads, houses, substations (for setbacks and cable distance)	OpenStreetMap via Overpass	✅ Free; coverage in rural TN/Gujarat is uneven
Wake physics	PyWake / FLORIS	✅ Open source
Locations and age of existing Indian turbines	OSM turbine points; a University of Leeds dataset; TheWindPower.net	⚠️ This is the gap

The gap in detail:

OSM turbine data isn’t comprehensive, because volunteers add it ad hoc. It has positions only, with no type or rating.
Leeds has a CC-BY dataset of Indian wind farms, built for a peer-reviewed paper on repowering India’s ageing wind farms. That’s both a citation and data, but it’s at farm level, not per turbine.
TheWindPower.net is paid, and most of its Indian coordinates are approximate.

Workaround for today: pick one cluster, say Muppandal or around Tenkasi. Pull whatever OSM turbine points exist there. Fill gaps by hand from satellite view; 30–50 turbines takes about 30 minutes. Assume the old turbines are about 500 kW at 50 m hubs. CEEW notes that pre-2000 turbines at India’s best sites are mostly below 500 kW with hubs of 50 m or lower. State that assumption on screen. Judges accept clearly labelled assumptions.

Is it a problem judges will find worth solving?

The case for it:

The government has formally named the problem. NIWE estimates 25.4 GW of repowering potential.
Agencies are tasked with exactly what your tool does: identifying turbines to repower.
There is real evidence of a stall. Repowering at the best sites has been slow even after the December 2023 policy revision.
You have academic backing (the Leeds paper) and a GIZ/NIWE report on repowering.

The risk, honestly:

Judging audience. The criteria ask “what changes for the people living with it?” and whether people living with the problem can use the fix. Repowering is a government and developer tool, which is less visceral than AQI or floods. You need to translate it into human terms in the first 20 seconds of the video:
More clean power from land that’s already a wind farm.
No new land acquisition from farmers.
X homes powered and Y tonnes of CO₂ avoided.
Your tool doesn’t solve the real blocker. Repowering is held back largely because old projects sit on 11 kV lines and many wind farms have multiple turbine owners within the same site.
Don’t claim to fix that.
Frame it as: “We give the state agency the evidence it needs to prioritise which clusters are worth the coordination effort.”
You could add a “fragmentation” note: owner count and 11 kV connection as manual inputs that lower the priority score. That shows judges you understand the real problem.
My verdict

Go ahead if you can scope it to one cluster:

a heatmap,
old layout vs optimised layout,
sliders,
an energy-gain number.

It’s past 2:45 PM Saturday. If the turbine-location gap feels like too much, the fallback with zero data risk is new-site screening: a suitability heatmap plus an optimal layout inside a polygon the user draws. It’s the same code minus the “existing turbines” layer. You lose the repowering story but keep everything else.

Want me to write the NASA POWER + GWA + PyWake pipeline for one cluster now so you can start running it?

Sources:

World Bank: GWA GIS data
Leeds: Indian wind farms dataset
EMD: OSM turbines
TheWindPower: India database
CEEW: Wind repowering
GIZ/NIWE: Repowering in India
Energetica India: Repowering policy
JMK Research: Repowering challenges
WeMakeDevs: Judging

so our solution is just telling clusters where govt should focus on solving , and if we are finding clusters then are we also giving info that the turbnes used there are old or new

Not quite. “Where should the government focus” is only the first of three outputs:

Where: a ranked list of clusters, with a priority score.
What to build there: an optimised new layout (how many modern turbines, where, at what spacing and hub height).
What it’s worth: the energy gained per year, CO₂ avoided, homes powered, and a rough payback period, plus what-if sliders.

So it’s a decision tool: “Repower Cluster A first. Here is the layout, and it yields +X GWh/year.” It isn’t just a map with red dots. That’s the difference judges will notice.

Do we know whether the turbines are old or new?

Not directly from free data. No free source gives per-turbine age for India, so you have three ways to handle it.

A. Estimate age from turbine spacing (best for today, and it’s real ML).
Old and new turbines differ in size, and size shows up in spacing:

Old sub-MW machines have small rotors, roughly 40–50 m, so they’re packed closely together.
Modern 2.5–3.5 MW machines have rotors of 120 m or more and stand hundreds of metres apart.

From OSM turbine points, compute each turbine’s nearest-neighbour distances and local density. Then classify it as likely old (<1 MW), mid (1–2 MW) or modern (>2 MW). Train or calibrate the classifier on the few points that do carry capacity tags, and on farms whose commissioning year is known from public news.

This is a heuristic, so label it on screen as “estimated turbine class, confidence X%”.

B. Detect turbines from satellite imagery.
An object detector finds turbines and estimates rotor size from the image. Researchers have done this, but it’s too much for the next 24 hours. Mention it in the video as future scope.

C. Let the government upload its own data.
State nodal agencies already hold turbine registries with capacity and commissioning year, because they handle repowering applications. Your app accepts a CSV upload of turbine ID, coordinates, capacity and year, and overrides the estimates when one is given.

This is the honest pitch: “We estimate from public data for the demo; in deployment, the state plugs in its registry and the engine gives exact results.”

Do A + C. The map then colours each turbine by estimated class (old/mid/modern), and the cluster score is driven mainly by how many old turbines a cluster has and how much better its wind is at 120–150 m hub height.

Want me to write the spacing-based classifier and cluster-scoring code now?

before writing code tell me the complete plan in a file in such a way so that i can solve it today only

I’ll put the full build plan into a document you can follow step by step today.

Used Claude Docs integration, loaded a skill

Outline’s up; now filling the goal and scope.

Used Claude Docs integration

O

Used artifacts

Filling goal and scope first.

Edit doc

Now the problem statement with citable facts.

Edit doc

Next: data sources.

Edit doc

Now the ML and optimisation modules.

Edit doc

Now the hour-by-hour timeline.

Edit doc

Now the frontend screens.

Edit doc

Next: video script and submission checklist.

Edit doc

Now risks and fallbacks.

Edit doc

The prose sections are in; last is the architecture diagram.

Make doc changes (batch)

Quick check of how the diagram renders.

Loaded tools, read a file, used Claude Docs integration

PawanSthal — One-Day Build Plan is ready. You can edit it inline or comment, and I’ll revise. I left one question in the doc about your team size and which demo cluster to use.

can you explain the govt scheme in a little more detail

Read a page

The policy is the National Repowering & Life Extension Policy for Wind Power Projects – 2023, issued by MNRE. It replaced the 2016 repowering policy. I read the full text; here’s what it says and how it maps onto PawanSthal.

Why it exists

Most turbines installed in India before 2000 are below 1 MW and sit on the best wind sites. Many have finished or are near the end of their design life. They also stand on short towers, with hubs of 30–60 m, while new turbines go up at 120–140 m. The policy lets owners replace them with modern machines before their design life ends, or refurbish them to extend their life, subject to safety checks.

Its stated goal is to maximise energy yield (kWh) per square kilometre of project area using the latest onshore technology. That is exactly what your optimiser computes.

Who is eligible

A turbine qualifies if any one of these holds:

it doesn’t meet MNRE’s quality control order;
it has completed its certified design life;
it is rated below 2 MW;
it is more than 15 years old and the owner chooses to repower for commercial reasons.
The key test: 1.5×

A repowered project must generate at least 1.5 times the old project’s actual generation. The baseline is the average of the last 3 years before disconnection, and the new project has 3 years after commissioning to meet the target.

For your app: show a “passes the 1.5× policy test: yes/no” badge for each cluster. Judges will see you’ve read the actual policy.

Two kinds of projects
Standalone: one owner. The owner submits a Detailed Project Report (DPR) to the State Nodal Agency (SNA). Alternatively, the SNA identifies candidate turbines itself and invites the owner to submit one.
Aggregation: many owners sharing infrastructure, which is the common case in old farms. The SNA or CNA identifies the turbines, then appoints a Wind Repowering Project Aggregator (WRPA). The aggregator can be a state or central PSU, a private developer chosen transparently, or one of the existing owners who has consent letters from all the others. The aggregator buys out the old assets, decommissions them and disposes of the blades, then rebuilds the site as if it were a greenfield project.

SNAs handle turbines connected to the state grid (InSTS). The Central Nodal Agency handles those on the inter-state grid (ISTS).

The process
The SNA or CNA receives or prepares the DPR.
It checks transmission capacity with the transmission utility; the developer must give 2 years’ notice for more capacity.
It gets the local DISCOM’s in-principle consent and issues a consent letter.
The Wind Repowering Committee grants a pause of up to 2 years on supplying power under the PPA while repowering happens.
The project must be commissioned within 24 months of the consent letter. Partial commissioning is allowed.
What happens to the power
The DISCOM keeps buying the old average generation under the existing PPA. The PPA is extended by the repowering period, capped at 2 years.
The extra power belongs to the developer, who can sell it on the exchange or through new PPAs. This is the main financial incentive.
DISCOMs must get at least 1 year’s notice before work starts.
Incentives
Loans from REC, PFC and IREDA on the same terms as new projects.
An extra 0.25% interest rebate from IREDA.
All fiscal benefits available to new wind projects.
Freedom to re-site turbines for optimal output. The policy explicitly lets developers change turbine positions, which justifies your layout optimiser.
Governance

The Wind Repowering Committee is chaired by MNRE’s Joint Secretary (Wind), with NIWE as member secretary. It also includes IREDA, SECI, the central transmission utility and three independent experts. It monitors projects and recommends fixes.

Two clauses that matter most for your pitch

1. The data mandate. NIWE must build and maintain a database of all old projects: ownership, technology, turbine details, connected substation and land ownership. It must also analyse performance data to identify poor-performing sites for repowering, with SNAs and SLDCs supplying the data. That is literally PawanSthal’s job.

Pitch line: “The policy asks NIWE and SNAs to identify poor-performing sites. We built the analytics layer for that mandate.”

2. The valuation formula (Annex). The policy gives a formula for compensating old owners, called the Repowering Site Preparation Cost:

RSPC = Terminal Valuation + Decommissioning Cost − Scrap Income
Terminal Valuation = PPA Value + PPA Termination Incentive (~10%)
PPA Value = NPV of (tariff × generation − O&M) over the remaining PPA years

Its worked example values a 1 MW turbine with a 12.3% PLF at a ₹3/kWh tariff. If you implement this, your money audit uses the government’s own method, which is very strong for judges. Use the example’s PLF and tariff as defaults for old turbines.

State-wise repowering potential (MW, turbines below 2 MW)
State	Potential (MW)
Tamil Nadu	7,386.5
Gujarat	4,665.35
Maharashtra	3,431.35
Karnataka	3,023.35
Rajasthan	2,933.9
Andhra Pradesh	2,366.2
Madhya Pradesh	1,562
Kerala	28

Tamil Nadu has the largest share, which supports choosing Muppandal as your demo cluster. The policy also says NIWE will publish a repowering potential map; your app is effectively a working prototype of that map.

What the policy doesn’t solve

It sets up a process but still depends on getting consent from every owner in an aggregation project and on transmission upgrades. Those are the blockers behind slow uptake. Keep the honest framing: your tool tells agencies which clusters are worth that effort, and what each one yields.

Want me to add a “Policy hooks” section to the plan doc covering the 1.5× test, the RSPC formula and the NIWE database clause?

Source: MNRE, National Repowering & Life Extension Policy for Wind Power Projects – 2023 (PDF)


