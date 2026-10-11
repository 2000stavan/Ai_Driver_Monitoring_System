# Recording protocol — targeted Indian RHD data (v1)

**Why we record at all:** public Indian datasets cover fatigue, gaze, road scenes and everyday driving, but **not**
(a) phone use, eating and drinking in Indian right-hand-drive (RHD) cabins with per-driver labels, and
(b) **labelled** harsh-driving events from Indian drives (Pune VIIT and LEVIN have no event labels).
These sessions fill exactly those two gaps (INITIALPLAN §8, PLAN §5).

> **Safety rule, no exceptions:** no distraction, eating, drinking, phone use or drowsiness is ever performed while the car is moving on a road. Distraction is acted **in a parked car**. Harsh manoeuvres happen **only in an empty, closed space at low speed** with a licensed driver.

---

## 0. Before any recording
- [ ] Ask the college whether **ethics approval** is needed for recording volunteers; follow its process if yes.
- [ ] Every participant reads and signs [`CONSENT_FORM.md`](CONSENT_FORM.md) (print it). Keep signed forms offline and separate from the data.
- [ ] Assign each participant an ID (`P01`, `P02`, …). Never put names in file names.
- [ ] For Session B: written permission from the owner of the empty lot / ground.

## 1. Equipment
| Item | Use | Notes |
|---|---|---|
| Phone 1 (**cabin**) | Driver video | Windscreen / dashboard mount in front of the driver (RHD side). **Wide angle, slightly low**, so face **and lap** are in view (texting happens at lap height; our DMD tests showed a face-only view misses it). 1080p, 30 fps, landscape. Lock exposure if possible. |
| Phone 2 (**road**, Session B/C) | Road video | Behind the windscreen facing forward. Optional for Session A. |
| Phone 1 or 3 (**sensors**) | IMU + GPS | **Sensor Logger** app (iOS / Android, free): accelerometer, gyroscope, gravity, GPS; export CSV. Phone **fixed** (mount, not loose in a cup holder). |
| Second person | Operator + log keeper | Notes times on the log sheet (§5), gives cues, watches safety. |
| Props | Session A | Water bottle, cup / chai glass, a snack (biscuits, banana or a roll), the participant's own phone. |

## 2. Session A — parked-car distraction clips (~25 min per participant)
Car **parked, engine off**, in a safe spot. The participant sits in the driver's seat, hands on the wheel between actions.

**Start:** begin recording on all phones → operator **claps once in view of the cabin camera** and says "P01, session A, start" (sync + label marker). Then 10 s of normal posture looking ahead (**personal calibration**).

| # | Action (label) | Variations | Reps | ~Duration |
|---|---|---|---|---|
| 1 | **Eating** (`eating`) — priority 1 | snack with one hand · with both hands · banana / roll | 3 each | 15–20 s |
| 2 | **Phone call** (`phone_call`) | phone at **right** ear · at **left** ear | 2 each | 20–30 s |
| 3 | **Texting / scrolling** (`texting`) | phone at lap · phone at wheel height · one hand · both hands | 2 each | 15–20 s |
| 4 | **Drinking** (`drinking`) | bottle · cup · chai glass | 2 each | 10–15 s |
| 5 | Normal driving posture (`safe`) | looking ahead, hands on wheel | 4 | 20 s |
| 6 | Mirror checks (`safe`, `mirror_check`) | left mirror · right mirror · rear-view | 2 each | 5 s |
| 7 | Dashboard / radio (`other`) | touch screen / radio | 2 | 10 s |
| 8 | **Hard negatives** (`other`) | adjust glasses · touch face · scratch chin · hand on head · rub eyes | 2 each | 5–10 s |

Between actions: 5 s of normal posture. The operator **says the action name aloud** before each one and notes the start time on the log sheet.

**Conditions:** repeat the full list in **daylight**; if possible a shorter repeat at **dusk / night** (interior light off, street light only) and with **harsh glare** (sun on the face). Record glasses / sunglasses / head covering / beard **as the participant normally is**; never ask anyone to change appearance.

**Hand use:** note which hand is used for each action (`left` / `right`); the car is always RHD.

## 3. Session B — controlled harsh-driving events (~20 min, one driver at a time)
**Empty, closed space only** (college ground, empty parking lot). No people or vehicles nearby. Licensed driver, seatbelts on, operator in the passenger seat. Max **30 km/h**.

Phones: sensor phone fixed; cabin + road phones recording. **Start:** clap in front of the cabin camera + operator says "P01, session B, start". Then **60 s of normal slow driving** (orientation calibration for the IMU).

| # | Manoeuvre (label) | How | Reps |
|---|---|---|---|
| 1 | **Harsh brake** (`HARSH_BRAKE`) | reach ~25 km/h, brake firmly to a stop | 8 |
| 2 | Normal brake (`normal_brake`) | reach ~25 km/h, brake gently | 5 |
| 3 | **Harsh acceleration** (`HARSH_ACCEL`) | from stop, accelerate hard to ~25 km/h | 8 |
| 4 | Normal acceleration (`normal_accel`) | from stop, accelerate gently | 5 |
| 5 | **Sharp turn** (`SHARP_TURN`) | ~15–20 km/h, tight left turn; then tight right turn | 5 each side |
| 6 | Normal turn (`normal_turn`) | gentle turn at low speed | 4 |
| 7 | **Speed bump** (if the space has one) (`speed_bump`) | cross at ~15 km/h | 4 |

The operator **says the manoeuvre aloud** just before it and writes the time. The non-harsh versions are just as important: they are the negatives the classifier learns from.

## 4. Session C — normal synced drive (optional, ~20–30 min)
Ordinary, legal, attentive driving on city roads with all three phones recording. **No acted events.** Used for: false-alarm rates in real Indian traffic, Indian normal-driving distributions, and context-attribution examples (lane changes, braking for real hazards).

## 5. Log sheet (one per session)
| Participant | Session | Date | Time of start clap | Car model | Phone models | Light (day/dusk/night/glare) | Glasses / head covering |
|---|---|---|---|---|---|---|---|

| # | Clock time (hh:mm:ss) | Action / manoeuvre | Hand (L/R) | Notes (missed, repeated, …) |
|---|---|---|---|---|

Use the **cabin phone's clock** for times (or a stopwatch started at the clap).

## 6. After the session
1. Copy files to the laptop as `P01_A_day_cabin.mp4`, `P01_B_road.mp4`, `P01_B_sensors.zip`, `P01_A_log.csv`.
2. Put them in `data/raw/own/<P01>/` (git-ignored) and back up to Drive (`My Drive/DMS/raw/own/`).
3. Never upload raw videos anywhere public.

## 7. Labelling (afterwards)
- **Session A:** CVAT or Label Studio, temporal segments for `eating / phone_call / texting / drinking / safe / mirror_check / other`; the log sheet gives the rough times, and pre-labels from YOLO-World / Grounding DINO speed it up.
- **Session B:** event times from the log sheet + the IMU peak, refined by looking at the sensor trace.
- **Agreement:** a second person labels ~10 % of Session A; report temporal IoU and segment-level F1 (INITIALPLAN §8).

## 8. Targets
| | Minimum (prototype) | Good (v1 target) |
|---|---|---|
| Participants (Session A) | 8 | 20–30 |
| Cars | 1 | 2–3 (hatchback, sedan, SUV) |
| Session B drivers | 2 | 4–5 |
| Session C | — | 2–4 h total |
