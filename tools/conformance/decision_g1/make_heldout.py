#!/usr/bin/env python3
"""Write G1's held-out decision cases from a fixed recipe: 40 slots, a literal seed, no resampling.

The held-out set is a targeted regression suite for gate G1, not a statistical sample: 40 requests (44 questions)
whose structure is fixed below, slot by slot, and whose content comes from a topic pool disjoint from the
representative set's (tools/conformance/decision_hf/make_cases.py). The recipe -- this file, its seed and its slot
table -- is committed before any representative F16 observation exists, and the request bytes are generated only
after G1's bounds are frozen (integration-tests/fixtures/decision/g1/bounds.json). Nothing here reads a logit, a
probability or an answer: no case is ever chosen, kept or replaced because of a numeric outcome. A change to the
recipe is a new held-out set.

Randomness comes from SplitMix64 seeded with SEED, through `below` (rejection sampling), so the output depends on this
file and the seed only, not on Python's `random` module or its version.

Prompt lengths. Six slots name a target length in prompt tokens: 63, 64 and 65 (around the CPU flash-attention
tiling boundary at 64 query rows), 600, 1,500 and 4,064 (near the 4,096-token default budget). The length is measured
with upstream's own preparation: `decision-oracle run --vocab-only` on the pinned GGUF's vocabulary
(tools/conformance/decision), which renders and tokenizes the request without evaluating anything. Padding is
deterministic (`pad`): first the most of the slot's filler sentences, in order, that keep the prompt within the
target, then words from FILLER_WORDS, placed at the end of a text state or before the final period of an object
state's last note. The words come from a breadth-first search over increasing-index subsequences of FILLER_WORDS:
each level (at most MAX_WORD_LEVELS) appends one later word to every sequence the previous level kept, measures them
all in one batch and keeps the first sequence found for each new prompt length within the target, until one reaches
the target exactly; the search is tried with the most sentences, then one and two fewer. The padded request is then
measured again and must be exactly the target. If no exact hit results, generation fails: the recipe is fixed and
re-reviewed; there is no retry with another seed.

The shared domain. Every request must build the same prompt on the HF reference and in llama.cpp
(tools/conformance/decision_hf/DIFFERENCES.md); `check_domain` enforces the rules executably and the comparator
re-checks the outcome (an identity mismatch fails the held-out run; it never becomes an exclusion or a replacement):

- question ids and choice keys match ^[a-z][a-z0-9_]{0,31}$, or are single ASCII letters in the letter-key slots;
- instructions, descriptions and score levels are non-blank strings, with no surrounding whitespace;
- no '<' or '>' anywhere, so no CONTROL or USER_DEFINED token spelling;
- JSON states hold string keys, strings, booleans, null, integers in signed i64 and floats with at most 6
  significant digits in [1e-4, 1e15) or 0; no duplicate keys;
- question types choice, noul and score only; no `multiple`, typed questions, media, `files` or `model`; noul
  criteria are absent or an object of non-blank strings under `true` and/or `false`.

Usage:
  make_heldout.py --oracle BUILD/bin/decision-oracle --vocab d1-2B-Q4_K_M.gguf [--out DIR]
  make_heldout.py --check-domain DIR      # check case files against the domain rules
  make_heldout.py --recipe                # print the slot table and the recipe digest

Writes DIR/cases/<id>.json (case schema v1), DIR/allowlist.json (every case included) and DIR/generation.json (the
recipe digest, the seed, the measuring tool's identity, every request digest and measured length). Standard library
only; Python 3.9+.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUT = REPO / "integration-tests" / "fixtures" / "decision" / "g1" / "heldout"

SEED = 0x5EED6011D1
PROFILE = "lfm3-d1/text/v1"
ARTIFACT = "d1-2b-f16"
SETTINGS = "cpu-t4-fa-off"  # the collection run evaluates every case under both G1 settings

MASK = (1 << 64) - 1


class SplitMix64:
    """Steele, Lea and Flood's SplitMix64: a fixed, documented sequence for a given seed."""

    def __init__(self, seed: int):
        self.state = seed & MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Uniform in [0, n), by rejection."""
        if n <= 0:
            raise ValueError("below(n) needs n > 0")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.next()
            if x < limit:
                return x % n

    def pick(self, items: Sequence[Any]) -> Any:
        return items[self.below(len(items))]

    def sample(self, items: Sequence[Any], k: int) -> List[Any]:
        """k distinct items, in the order drawn."""
        pool = list(items)
        out = []
        for _ in range(k):
            out.append(pool.pop(self.below(len(pool))))
        return out

    def shuffled(self, items: Sequence[Any]) -> List[Any]:
        return self.sample(items, len(items))


# ---------------------------------------------------------------------------------------------------------------------
# Topics: none of them is the representative set's (support tickets for billing/shipping, planets, languages, reviews)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Topic:
    name: str
    texts: List[str]
    short: str
    choice: str
    options: List[Tuple[str, str]]
    nouls: List[Tuple[str, str, str]]
    score: Tuple[str, List[str]]
    entity: str
    statuses: List[str]
    filler: List[str]
    unicode_text: str = ""


TOPICS: List[Topic] = [
    Topic(
        name="clinic",
        texts=[
            "A 34-year-old runner reports knee pain that started after a trail race two days ago and gets worse on "
            "stairs.",
            "An elderly patient has had a dry cough for three weeks, sleeps poorly and lost some appetite, but has no "
            "fever.",
            "A parent calls because their child has a rash on both arms after a weekend at the lake; the child is "
            "playing normally.",
        ],
        short="Mild headache since this morning.",
        choice="Which service should see this patient first?",
        options=[
            ("orthopedics", "Bones, joints and sports injuries"),
            ("dermatology", "Skin conditions and rashes"),
            ("pulmonology", "Lungs and breathing"),
            ("general_practice", "Routine assessment and referral"),
            ("pediatrics", "Children and adolescents"),
            ("emergency", "Immediate danger to life"),
            ("physiotherapy", "Rehabilitation and movement"),
            ("pharmacy", "Medication advice only"),
        ],
        nouls=[
            ("Does the patient need to be seen today?", "Symptoms could get worse within hours",
             "It can safely wait for a routine appointment"),
            ("Is a fever mentioned?", "The text reports a raised temperature", "No fever is reported"),
        ],
        score=("How urgent is this case?", ["no urgency", "very low", "low", "slight", "moderate", "notable", "high",
                                            "very high", "severe", "critical"]),
        entity="patient",
        statuses=["waiting", "triaged", "referred", "discharged"],
        filler=[
            "The patient arrived at the front desk at nine and filled in the intake form.",
            "Blood pressure was measured twice, sitting and standing.",
            "The nurse noted that the patient drinks two coffees a day and does not smoke.",
            "No allergies to medication were declared.",
            "The previous visit was eleven months ago, for a vaccination.",
            "The patient walks to work and climbs stairs every day.",
            "Sleep is described as regular, about seven hours a night.",
            "There is no family history of heart disease that the patient knows of.",
            "The waiting room was quiet and the triage took ten minutes.",
            "The patient asked whether a sick note would be needed for work.",
        ],
        unicode_text="Patient: Mme Hélène Brändli, née à Zürich; douleur au genou après la course, pas de fièvre.",
    ),
    Topic(
        name="insurance",
        texts=[
            "A storm broke a window and water soaked the living room carpet; the policy holder sent photos the same "
            "night.",
            "A parked car was scratched along one side in a supermarket car park; there are no witnesses.",
            "A laptop was stolen from a train while its owner slept; the theft was reported to the police the next "
            "morning.",
        ],
        short="My phone screen cracked yesterday.",
        choice="Which kind of claim is this?",
        options=[
            ("home_damage", "Damage to a building or its contents"),
            ("vehicle", "Damage to or theft of a car"),
            ("theft", "Stolen personal belongings"),
            ("liability", "Damage caused to someone else"),
            ("travel", "Problems during a trip abroad"),
            ("health", "Medical costs"),
            ("legal_cover", "Legal advice and disputes"),
            ("not_covered", "Nothing the policy covers"),
        ],
        nouls=[
            ("Was the event reported to the police?", "A police report exists or was filed",
             "No police report is mentioned"),
            ("Did the damage happen at home?", "The damage is at the insured address",
             "It happened somewhere else"),
        ],
        score=("How likely is the claim to be paid in full?", ["impossible", "very unlikely", "unlikely", "doubtful",
                                                              "uncertain", "plausible", "likely", "very likely",
                                                              "almost certain", "certain"]),
        entity="claim",
        statuses=["opened", "assessing", "approved", "declined"],
        filler=[
            "The policy was renewed in March without changes.",
            "The deductible on this contract is two hundred euros.",
            "An adjuster may visit within ten working days.",
            "The policy holder has made one earlier claim, five years ago.",
            "Receipts for the damaged items were attached as scans.",
            "The claim form was signed electronically.",
            "The customer prefers to be contacted by email in the afternoon.",
            "A reference number was sent by text message.",
            "The bank details for any payment were confirmed.",
            "The customer asked how long the assessment usually takes.",
        ],
    ),
    Topic(
        name="allergens",
        texts=[
            "The recipe uses ground almonds, butter, eggs and a pinch of salt; the glaze is made with lemon juice.",
            "This curry is cooked with coconut milk, lentils, cumin and fresh coriander, served with rice.",
            "The bread contains wheat flour, sesame seeds, yeast and olive oil, and is baked in a shared oven.",
        ],
        short="Plain rice with steamed carrots.",
        choice="Which allergen is the main concern in this dish?",
        options=[
            ("tree_nuts", "Almonds, walnuts, hazelnuts and similar"),
            ("gluten", "Wheat, barley and rye"),
            ("dairy", "Milk, butter and cheese"),
            ("eggs", "Whole eggs or egg products"),
            ("sesame", "Sesame seeds or oil"),
            ("shellfish", "Shrimp, crab and lobster"),
            ("soy", "Soy beans and soy sauce"),
            ("none_listed", "No common allergen"),
        ],
        nouls=[
            ("Is the dish safe for someone with a nut allergy?", "No nuts or nut traces are involved",
             "Nuts or traces of nuts could be present"),
            ("Is the dish vegan?", "It contains no animal products", "It contains meat, eggs, dairy or honey"),
        ],
        score=("How risky is this dish for a guest with severe allergies?", ["no risk", "negligible", "very low",
                                                                             "low", "some", "moderate", "high",
                                                                             "very high", "dangerous",
                                                                             "certain exposure"]),
        entity="dish",
        statuses=["draft", "tested", "on_menu", "retired"],
        filler=[
            "The kitchen opens at seven and the first batch is prepared before lunch.",
            "Every ingredient label is kept in a binder near the pass.",
            "The same knives are washed between preparations.",
            "Portions are weighed before they leave the kitchen.",
            "The menu changes with the season, every three months.",
            "Staff were trained on the allergen chart last spring.",
            "Deliveries arrive on Tuesday and Friday mornings.",
            "The pastry station shares a fridge with the salad station.",
            "Leftovers are labelled with the date and discarded after two days.",
            "Guests are asked about allergies when they book a table.",
        ],
    ),
    Topic(
        name="travel",
        texts=[
            "The connecting flight to Lisbon was cancelled after boarding closed, and the next one leaves tomorrow "
            "afternoon.",
            "The hotel booked for three nights has no record of the reservation, although a confirmation email "
            "arrived last week.",
            "A rental car was returned with a full tank, but the company charged a refuelling fee.",
        ],
        short="My train is delayed by ten minutes.",
        choice="What should the traveller ask for first?",
        options=[
            ("rebooking", "A seat on another flight or train"),
            ("refund", "Money back for the unused service"),
            ("hotel_voucher", "A room for the night"),
            ("compensation", "A payment for the delay"),
            ("fee_reversal", "Removal of an unjustified charge"),
            ("new_booking", "A fresh reservation elsewhere"),
            ("written_proof", "A written statement of what happened"),
            ("nothing", "No request is needed"),
        ],
        nouls=[
            ("Is the traveller stranded overnight?", "They cannot continue the trip today",
             "They can still travel today"),
            ("Was the problem caused by the company?", "The provider is at fault",
             "The traveller or an outside event caused it"),
        ],
        score=("How disruptive is this for the trip?", ["none", "trivial", "minor", "small", "noticeable",
                                                       "significant", "serious", "major", "severe",
                                                       "trip ruined"]),
        entity="booking",
        statuses=["confirmed", "changed", "cancelled", "completed"],
        filler=[
            "The traveller checked in online the evening before.",
            "Luggage was limited to one cabin bag.",
            "The boarding pass was stored on a phone.",
            "The airport was busy because of a holiday weekend.",
            "Announcements were made in three languages.",
            "The traveller has a loyalty card with the airline.",
            "Travel insurance was bought together with the ticket.",
            "The booking was paid with a credit card.",
            "A seat by the window had been reserved.",
            "The return journey is planned for the following Sunday.",
        ],
        unicode_text="Reisende: Frau Jürgensen aus Göteborg; der Anschlussflug nach Málaga fällt aus, Hotel in São "
                     "Paulo bestätigt.",
    ),
    Topic(
        name="hr_leave",
        texts=[
            "An employee asks for three days off next week to move house, and has twelve days of leave left this "
            "year.",
            "A team lead reports that a colleague has been off sick for two weeks without sending a doctor's note.",
            "A new parent wants to switch to four days a week for six months after their leave ends.",
        ],
        short="Can I take Friday off?",
        choice="Which policy applies to this request?",
        options=[
            ("annual_leave", "Paid holidays from the yearly allowance"),
            ("sick_leave", "Absence because of illness"),
            ("parental_leave", "Leave for a new child"),
            ("flexible_work", "Changes to hours or working pattern"),
            ("unpaid_leave", "Time off without pay"),
            ("special_leave", "Moving, weddings, bereavement"),
            ("training", "Courses and exams"),
            ("not_a_leave_request", "Something other than time off"),
        ],
        nouls=[
            ("Does the request need a manager's approval?", "A manager must sign it off",
             "It is granted automatically by policy"),
            ("Is a document required?", "A certificate or form must be provided", "No paperwork is needed"),
        ],
        score=("How much planning does the team need for this absence?", ["none", "minimal", "very little",
                                                                         "little", "some", "moderate",
                                                                         "considerable", "a lot", "extensive",
                                                                         "a full replacement"]),
        entity="request",
        statuses=["submitted", "approved", "rejected", "withdrawn"],
        filler=[
            "The office is closed on public holidays.",
            "Leave requests are usually answered within three working days.",
            "The team has seven members across two sites.",
            "Shift swaps are arranged on a shared calendar.",
            "The handbook was updated at the start of the year.",
            "Overtime is recorded in the time-tracking tool.",
            "Managers review absences once a month.",
            "New starters receive their allowance pro rata.",
            "The payroll cut-off is the twentieth of each month.",
            "Questions can be sent to the people team mailbox.",
        ],
    ),
    Topic(
        name="contracts",
        texts=[
            "Either party may end this agreement with thirty days' written notice, without giving a reason.",
            "The supplier is not liable for indirect losses, including lost profits, under any circumstances.",
            "All disputes shall be settled by arbitration in Geneva, under the rules in force at the time.",
        ],
        short="Payment is due within 14 days.",
        choice="What kind of clause is this?",
        options=[
            ("termination", "How the agreement ends"),
            ("liability", "Limits on who pays for losses"),
            ("dispute_resolution", "Courts, arbitration and governing law"),
            ("payment", "Prices, invoices and deadlines"),
            ("confidentiality", "Keeping information secret"),
            ("intellectual_property", "Ownership of work and rights"),
            ("force_majeure", "Events outside anyone's control"),
            ("definitions", "Meanings of terms used"),
        ],
        nouls=[
            ("Does the clause favour the supplier?", "It mainly protects the supplier",
             "It is neutral or protects the customer"),
            ("Does the clause set a deadline?", "A time limit is stated", "No time limit is stated"),
        ],
        score=("How risky is this clause for the customer?", ["no risk", "negligible", "very low", "low",
                                                             "moderate", "noticeable", "high", "very high",
                                                             "severe", "unacceptable"]),
        entity="clause",
        statuses=["proposed", "negotiated", "agreed", "struck"],
        filler=[
            "The draft was shared by the supplier's counsel on Monday.",
            "Comments are tracked in the margin of the document.",
            "The agreement runs for an initial term of two years.",
            "Annexes describe the services and the service levels.",
            "Signatures are collected through an electronic platform.",
            "Both parties keep a copy of the signed version.",
            "Headings are for convenience and do not change the meaning.",
            "Notices must be sent to the addresses in the first section.",
            "The customer's legal team reviews every renewal.",
            "Amendments are only valid in writing.",
        ],
    ),
    Topic(
        name="code_review",
        texts=[
            "The pull request renames a public function and updates every caller, but the changelog is not updated.",
            "A new loop reads a file line by line and keeps every line in memory before processing it.",
            "The change catches all exceptions around the network call and silently returns an empty list.",
        ],
        short="Fixes a typo in a comment.",
        choice="What is the main concern in this change?",
        options=[
            ("api_break", "A change that breaks existing users"),
            ("performance", "Speed or memory use"),
            ("error_handling", "Errors hidden or mishandled"),
            ("security", "Data exposure or injection"),
            ("tests", "Missing or weak tests"),
            ("documentation", "Docs or changelog out of date"),
            ("style", "Naming and formatting"),
            ("no_concern", "Ready to merge"),
        ],
        nouls=[
            ("Should the change be blocked until fixed?", "Merging it as is would cause harm",
             "It can be merged with follow-ups"),
            ("Does the change need new tests?", "Its behaviour is not covered by tests", "Existing tests cover it"),
        ],
        score=("How risky is merging this change?", ["no risk", "trivial", "very low", "low", "some", "moderate",
                                                    "high", "very high", "severe", "dangerous"]),
        entity="change",
        statuses=["open", "changes_requested", "approved", "merged"],
        filler=[
            "The continuous integration run finished in twelve minutes.",
            "Two reviewers were assigned automatically.",
            "The branch is three commits ahead of the main branch.",
            "Formatting was checked by the pre-commit hook.",
            "The author linked the issue in the description.",
            "A screenshot shows the new output of the command.",
            "The change touches four files in one module.",
            "Benchmarks were not run for this change.",
            "The reviewer left a comment on the naming of a variable.",
            "The release is planned for the end of the month.",
        ],
    ),
    Topic(
        name="it_incidents",
        texts=[
            "Users in the Berlin office cannot reach the intranet since 08:40, while the public website works.",
            "The database disk is 92 percent full and growing by one percent a day.",
            "A scheduled certificate renewal failed; the current certificate expires in nine days.",
        ],
        short="One printer is offline.",
        choice="Which team should own this incident?",
        options=[
            ("network", "Connectivity, routing and VPN"),
            ("database", "Database servers and storage"),
            ("security", "Certificates, access and threats"),
            ("desktop_support", "Laptops, printers and accounts"),
            ("platform", "Servers, containers and deployments"),
            ("application", "Bugs in business software"),
            ("vendor", "A third-party service provider"),
            ("no_action", "Not an incident"),
        ],
        nouls=[
            ("Is a customer-facing service affected?", "Customers notice the problem",
             "Only internal users are affected"),
            ("Is data at risk of being lost?", "Data could be lost or corrupted", "No data is at risk"),
        ],
        score=("How severe is this incident?", ["negligible", "very minor", "minor", "low", "moderate",
                                               "elevated", "major", "very major", "critical", "catastrophic"]),
        entity="incident",
        statuses=["new", "investigating", "mitigated", "resolved"],
        filler=[
            "Monitoring raised an alert that was acknowledged by the on-call engineer.",
            "The status page was not updated yet.",
            "The last change to this system was deployed on Tuesday.",
            "Logs are kept for thirty days.",
            "The runbook for this service was reviewed in spring.",
            "A bridge call was opened for the responders.",
            "The vendor's support line answered within an hour.",
            "Backups run every night at two o'clock.",
            "The affected office has about eighty employees.",
            "A post-incident review is scheduled for every major outage.",
        ],
    ),
    Topic(
        name="utilities",
        texts=[
            "The electricity bill for January is three times last January's, although nobody lived in the flat for "
            "two weeks.",
            "The water meter reading on the bill is lower than the reading the tenant took on the same day.",
            "The gas supplier announced a new tariff by letter, effective next month, with a higher standing "
            "charge.",
        ],
        short="My bill arrived late this month.",
        choice="What is the customer most likely asking for?",
        options=[
            ("bill_correction", "A corrected invoice"),
            ("meter_check", "A visit to read or test the meter"),
            ("tariff_change", "A cheaper or different tariff"),
            ("payment_plan", "Paying in instalments"),
            ("explanation", "An explanation of the charges"),
            ("cancellation", "Ending the contract"),
            ("refund", "Money back for overpayment"),
            ("no_request", "Just information, no action"),
        ],
        nouls=[
            ("Does the customer dispute the amount?", "They say the amount is wrong",
             "They accept the amount"),
            ("Is a meter reading mentioned?", "A reading is given or questioned", "No reading is mentioned"),
        ],
        score=("How likely is an error on the bill?", ["impossible", "very unlikely", "unlikely", "slightly possible",
                                                      "possible", "plausible", "likely", "very likely",
                                                      "almost certain", "certain"]),
        entity="account",
        statuses=["active", "in_review", "closed", "moving"],
        filler=[
            "The account has been active for four years.",
            "Payments are made by direct debit on the first of the month.",
            "The flat has electric heating and a gas cooker.",
            "The last meter reading was submitted online.",
            "Bills are sent by email as PDF files.",
            "The customer has a smart meter installed in the hall.",
            "Prices include taxes at the standard rate.",
            "The supplier's call centre is open on weekdays.",
            "An annual statement is issued every December.",
            "The tenant shares the flat with one other person.",
        ],
    ),
    Topic(
        name="schooling",
        texts=[
            "A pupil has missed six maths lessons this term and her last two tests were well below her usual marks.",
            "A parent asks whether their son can skip the swimming lessons because of an ear infection.",
            "Two students handed in essays with several identical paragraphs.",
        ],
        short="The bus was late this morning.",
        choice="Who should handle this first?",
        options=[
            ("class_teacher", "The pupil's main teacher"),
            ("school_nurse", "Health questions"),
            ("head_of_year", "Discipline and attendance"),
            ("learning_support", "Extra help with learning"),
            ("counsellor", "Wellbeing and personal issues"),
            ("exams_office", "Tests, marks and certificates"),
            ("parents", "The family"),
            ("no_one", "Nothing to handle"),
        ],
        nouls=[
            ("Should the parents be contacted?", "The family needs to know or act", "It can be handled at school"),
            ("Is a medical reason given?", "Illness or injury is mentioned", "No medical reason is given"),
        ],
        score=("How concerning is this situation?", ["not at all", "barely", "very slightly", "slightly", "somewhat",
                                                    "moderately", "quite", "very", "seriously", "extremely"]),
        entity="pupil",
        statuses=["enrolled", "on_watch", "supported", "excused"],
        filler=[
            "The school day starts at eight fifteen.",
            "Homework is posted on the class website every Friday.",
            "Parents' evening takes place twice a year.",
            "The class has twenty-four pupils.",
            "Lunch is served in two sittings.",
            "The library is open after school until five.",
            "Reports are sent home at the end of each term.",
            "The school runs a homework club on Wednesdays.",
            "Sports day is planned for early June.",
            "Teachers meet every Monday to plan the week.",
        ],
    ),
    Topic(
        name="property",
        texts=[
            "The tenant reports black mould on the bedroom ceiling that came back two weeks after it was cleaned.",
            "The landlord wants to raise the rent by 9 percent at the end of the fixed term.",
            "A neighbour's tree has roots that lifted the paving along the shared driveway.",
        ],
        short="The doorbell does not ring.",
        choice="What kind of issue is this?",
        options=[
            ("repair", "Something broken that the landlord must fix"),
            ("health_hazard", "Damp, mould, gas or electrical danger"),
            ("rent", "Rent level or payment"),
            ("deposit", "The security deposit"),
            ("neighbour_dispute", "A problem with neighbours"),
            ("contract", "Terms of the lease"),
            ("cleaning", "Cleaning or rubbish"),
            ("no_issue", "Nothing to resolve"),
        ],
        nouls=[
            ("Is the landlord responsible?", "The landlord must act or pay", "The tenant or a third party is"),
            ("Is there a risk to health?", "The problem could make people ill or hurt them",
             "There is no health risk"),
        ],
        score=("How urgent is the repair?", ["not urgent", "can wait months", "can wait weeks", "low", "routine",
                                            "soon", "urgent", "very urgent", "emergency", "dangerous now"]),
        entity="tenancy",
        statuses=["active", "notice_given", "ending", "renewed"],
        filler=[
            "The flat is on the third floor of a building from the seventies.",
            "The lease was signed for one year with an option to renew.",
            "Rent is paid by bank transfer before the fifth of the month.",
            "The deposit is held by a protection scheme.",
            "The building has a caretaker on weekday mornings.",
            "Heating is shared with the other flats.",
            "The windows were replaced three years ago.",
            "A smoke alarm is installed in the hallway.",
            "Bicycles are stored in the basement.",
            "The tenant has lived there since last September.",
        ],
    ),
    Topic(
        name="sports",
        texts=[
            "The home team led 2-0 at half time, conceded three goals in ten minutes and lost 2-3.",
            "The marathon runner finished in 2:31, beating her personal best by four minutes in windy conditions.",
            "The match was stopped in the second set after the player injured his ankle and could not continue.",
        ],
        short="The game starts at seven.",
        choice="How did the event end for the athlete or team?",
        options=[
            ("win", "A victory"),
            ("loss", "A defeat"),
            ("draw", "A tie"),
            ("retirement", "Stopped before the end"),
            ("personal_best", "A best-ever result"),
            ("disqualification", "Removed by the officials"),
            ("postponed", "Moved to another date"),
            ("unknown", "The text does not say"),
        ],
        nouls=[
            ("Did the athlete or team win?", "They won the event", "They did not win"),
            ("Was anyone injured?", "An injury is reported", "No injury is reported"),
        ],
        score=("How disappointing is the result for the fans?", ["delightful", "very pleasing", "pleasing",
                                                                "fine", "mixed", "a bit disappointing",
                                                                "disappointing", "very disappointing",
                                                                "painful", "devastating"]),
        entity="event",
        statuses=["scheduled", "live", "finished", "abandoned"],
        filler=[
            "The stadium was almost full despite the rain.",
            "Tickets had sold out three weeks in advance.",
            "The coach made two changes to the starting line-up.",
            "Temperatures dropped sharply in the evening.",
            "The event was broadcast on regional television.",
            "Supporters travelled from neighbouring towns.",
            "The referee was officiating his first final.",
            "Warm-ups started forty minutes before kick-off.",
            "The previous meeting between the two sides ended level.",
            "A minute of applause was held before the start.",
        ],
        unicode_text="Résumé: l'équipe de Saint-Étienne a mené 2-0, puis a perdu 2-3; le gardien Søren Ødegård "
                     "s'est blessé.",
    ),
]

TOPIC_BY_NAME = {t.name: t for t in TOPICS}

# The fixed scenarios: never drawn, never replaced.
AMBIGUOUS_CHOICE = {
    "state": "Patient reports chest tightness after a long run this morning; it eased after resting, came back "
             "briefly while climbing stairs, and there is no history of heart problems.",
    "question": {
        "type": "choice",
        "instructions": "Which service should see this patient first?",
        "criteria": {
            "cardiology": "Possible heart-related symptoms",
            "sports_medicine": "Exercise-related strain or injury",
            "general_practice": "Routine assessment and referral",
        },
    },
}
BORDERLINE_SCORE = {
    "state": "The nightly backup finished two hours late. No data was lost, but the morning sales report used "
             "yesterday's figures and two managers made decisions on them before the delay was noticed.",
    "question": {
        "type": "score",
        "instructions": "How severe is this incident?",
        "criteria": ["negligible", "minor", "moderate", "major", "critical"],
    },
}

RIVERS = [
    "danube", "rhine", "elbe", "loire", "seine", "rhone", "tagus", "douro", "ebro", "po", "tiber", "arno", "oder",
    "vistula", "dnieper", "volga", "don", "thames", "severn", "shannon", "meuse", "scheldt", "garonne", "dordogne",
    "inn", "drava", "sava",
]  # fmt: skip
SPICES = [
    "anise", "basil", "bay_leaf", "caraway", "cardamom", "cayenne", "chervil", "chives", "cinnamon", "clove",
    "coriander", "cumin", "dill", "fennel", "fenugreek", "galangal", "garlic", "ginger", "juniper", "lavender",
    "lemongrass", "mace", "marjoram", "mint", "mustard", "nutmeg", "oregano", "paprika", "parsley", "pepper",
    "rosemary", "saffron", "sage", "sumac", "tarragon", "thyme",
]  # fmt: skip
ADJECTIVES = ["red", "blue", "small", "large", "light", "heavy", "round", "square", "soft", "hard", "warm", "cool",
              "dark", "bright", "short", "long", "thin", "wide", "quiet", "loud"]  # fmt: skip
NOUNS = ["lamp", "chair", "table", "shelf", "mug", "plate", "towel", "rug", "clock", "vase", "basket", "pillow",
         "mirror", "frame", "jar", "bowl", "kettle", "fan"]  # fmt: skip
FILLER_WORDS = ["also", "noted", "today", "again", "later", "here", "then", "too", "still", "now", "soon", "please",
                "thanks", "indeed", "meanwhile", "overall", "however", "briefly", "certainly", "anyway"]  # fmt: skip

# ---------------------------------------------------------------------------------------------------------------------
# The slots: exact and in order
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Slot:
    number: int
    name: str
    kind: str  # choice | noul | score | multi
    size: int = 0  # options or levels
    keys: str = "words"  # words | upper | lower | rivers | spices | catalogue
    state: str = "text"  # text | object | array | null | empty | unicode
    criteria: str = "both"  # noul: both | none | true
    target: Optional[int] = None  # prompt tokens
    fixed: Optional[str] = None  # ambiguous | borderline
    title: str = ""
    covers: List[str] = field(default_factory=list)


SLOTS: List[Slot] = [
    Slot(1, "choice-2", "choice", 2, covers=["state-string", "choice-order"]),
    Slot(2, "choice-3", "choice", 3, covers=["state-string", "choice-order"]),
    Slot(3, "choice-4", "choice", 4, covers=["state-string", "choice-order"]),
    Slot(4, "choice-5", "choice", 5, covers=["state-string", "choice-order"]),
    Slot(5, "choice-ambiguous", "choice", 3, fixed="ambiguous", covers=["state-string", "choice-order"]),
    Slot(6, "choice-6", "choice", 6, covers=["state-string", "choice-order"]),
    Slot(7, "choice-8-600-tokens", "choice", 8, target=600, covers=["state-string", "choice-order"]),
    Slot(8, "choice-upper-letter-keys", "choice", 4, keys="upper", covers=["state-string", "choice-letter-keys"]),
    Slot(9, "choice-lower-letter-keys", "choice", 5, keys="lower",
         covers=["state-string", "choice-lowercase-keys"]),
    Slot(10, "choice-27", "choice", 27, keys="rivers", covers=["state-string", "choice-27"]),
    Slot(11, "choice-36-pool-fallback", "choice", 36, keys="spices", covers=["state-string", "choice-pool-fallback"]),
    Slot(12, "choice-255", "choice", 255, keys="catalogue", covers=["state-string", "choice-max-options"]),
    Slot(13, "choice-json-object", "choice", 4, state="object", covers=["state-object", "json-order"]),
    Slot(14, "choice-json-object-1500-tokens", "choice", 3, state="object", target=1500,
         covers=["state-object", "json-order"]),
    Slot(15, "choice-json-array", "choice", 4, state="array", covers=["state-array"]),
    Slot(16, "choice-null-state", "choice", 3, state="null", covers=["state-null"]),
    Slot(17, "choice-64-tokens", "choice", 2, target=64, covers=["state-string", "choice-order"]),
    Slot(18, "choice-4064-tokens", "choice", 3, target=4064, covers=["state-string", "budget"]),
    Slot(19, "noul-criteria-a", "noul", covers=["state-string", "noul-order"]),
    Slot(20, "noul-criteria-b", "noul", covers=["state-string", "noul-order"]),
    Slot(21, "noul-criteria-c", "noul", covers=["state-string", "noul-order"]),
    Slot(22, "noul-no-criteria", "noul", criteria="none", covers=["state-string", "noul-no-criteria"]),
    Slot(23, "noul-63-tokens", "noul", criteria="none", target=63, covers=["state-string", "noul-no-criteria"]),
    Slot(24, "noul-true-only", "noul", criteria="true", covers=["state-string", "noul-partial-criteria"]),
    Slot(25, "noul-json-object", "noul", state="object", covers=["state-object", "noul-order"]),
    Slot(26, "noul-json-array", "noul", state="array", covers=["state-array", "noul-order"]),
    Slot(27, "noul-empty-state", "noul", state="empty", criteria="none", covers=["state-empty", "noul-no-criteria"]),
    Slot(28, "noul-65-tokens", "noul", target=65, covers=["state-string", "noul-order"]),
    Slot(29, "score-2", "score", 2, covers=["state-string", "score-2-levels"]),
    Slot(30, "score-3", "score", 3, covers=["state-string"]),
    Slot(31, "score-borderline", "score", 5, fixed="borderline", covers=["state-string"]),
    Slot(32, "score-7", "score", 7, covers=["state-string"]),
    Slot(33, "score-10", "score", 10, covers=["state-string", "score-10-levels"]),
    Slot(34, "score-json-object", "score", 4, state="object", covers=["state-object"]),
    Slot(35, "score-null-state", "score", 3, state="null", covers=["state-null"]),
    Slot(36, "score-unicode", "score", 5, state="unicode", covers=["state-string", "json-unicode"]),
    Slot(37, "choice-unicode", "choice", 4, state="unicode", covers=["state-string", "json-unicode"]),
    Slot(38, "score-json-mixed", "score", 6, state="object", covers=["state-object", "json-int", "json-float",
                                                                     "json-literals"]),
    Slot(39, "multi-text", "multi", covers=["state-string", "multi-question"]),
    Slot(40, "multi-json", "multi", state="object", covers=["state-object", "multi-question"]),
]

# Topics of the unicode slots: the three with a non-ASCII text.
UNICODE_TOPICS = ["clinic", "travel", "sports"]


def case_id(slot: Slot) -> str:
    return f"g1h-{slot.number:02d}-{slot.name}"


# ---------------------------------------------------------------------------------------------------------------------
# Building requests
# ---------------------------------------------------------------------------------------------------------------------


def score_levels(topic: Topic, n: int) -> List[str]:
    scale = topic.score[1]
    return [scale[round(i * (len(scale) - 1) / (n - 1))] for i in range(n)]


def json_state(topic: Topic, rng: SplitMix64, mixed: bool) -> Dict[str, Any]:
    """A nested object with ordered, unsorted keys; with `mixed`, every JSON literal kind."""
    record: Dict[str, Any] = {
        "summary": rng.pick(topic.texts),
        topic.entity: {
            "status": rng.pick(topic.statuses),
            "id": 1000 + rng.below(9000),
            "tags": rng.sample([s for s in topic.statuses], 2),
        },
        "history": [
            {"step": 1, "note": rng.pick(topic.filler)},
            {"step": 2, "note": rng.pick(topic.filler)},
        ],
    }
    if mixed:
        record["metrics"] = {
            "count": rng.below(500),
            "ratio": [0.25, 0.5, 0.75, 0.125][rng.below(4)],
            "amount": 10 + rng.below(9000) / 4,
            "offset": -(1 + rng.below(1000)),
            "verified": rng.below(2) == 1,
            "reviewer": None,
            "large": 1_000_000_007 + rng.below(1000),
            "zero": 0.0,
        }
    return record


def json_array_state(topic: Topic, rng: SplitMix64) -> List[Any]:
    return [
        {"from": "first", "text": rng.pick(topic.texts)},
        {"from": "second", "text": rng.pick(topic.filler)},
        rng.pick(topic.statuses),
        1 + rng.below(99),
    ]


def choice_question(topic: Topic, slot: Slot, rng: SplitMix64) -> Dict[str, Any]:
    if slot.keys == "rivers":
        criteria = {name: f"The {name.replace('_', ' ')} basin office" for name in RIVERS[: slot.size]}
        return {"type": "choice", "instructions": "Which regional office should file this report?",
                "criteria": criteria}
    if slot.keys == "spices":
        criteria = {name: f"Shelf for {name.replace('_', ' ')}" for name in SPICES[: slot.size]}
        return {"type": "choice", "instructions": "Which shelf should this jar go on?", "criteria": criteria}
    if slot.keys == "catalogue":
        criteria = {}
        for i in range(slot.size):
            criteria[f"item_{i:03d}"] = f"{ADJECTIVES[i % len(ADJECTIVES)]} {NOUNS[(i * 7) % len(NOUNS)]}, lot {i}"
        return {"type": "choice", "instructions": "Which catalogue item does the request describe?",
                "criteria": criteria}
    options = rng.sample(topic.options, slot.size)
    if slot.keys == "upper":
        keys = [chr(ord("A") + i) for i in range(slot.size)]
    elif slot.keys == "lower":
        keys = [chr(ord("a") + i) for i in range(slot.size)]
    else:
        keys = [key for key, _ in options]
    return {"type": "choice", "instructions": topic.choice,
            "criteria": {key: desc for key, (_, desc) in zip(keys, options)}}


def noul_question(topic: Topic, criteria: str, rng: SplitMix64) -> Dict[str, Any]:
    instruction, yes, no = rng.pick(topic.nouls)
    question: Dict[str, Any] = {"type": "noul", "instructions": instruction}
    if criteria == "both":
        question["criteria"] = {"true": yes, "false": no}
    elif criteria == "true":
        question["criteria"] = {"true": yes}
    return question


def score_question(topic: Topic, n: int) -> Dict[str, Any]:
    return {"type": "score", "instructions": topic.score[0], "criteria": score_levels(topic, n)}


def build_request(slot: Slot, topic: Topic, rng: SplitMix64) -> Dict[str, Any]:
    if slot.fixed == "ambiguous":
        return {"state": AMBIGUOUS_CHOICE["state"], "questions": {"service": AMBIGUOUS_CHOICE["question"]}}
    if slot.fixed == "borderline":
        return {"state": BORDERLINE_SCORE["state"], "questions": {"severity": BORDERLINE_SCORE["question"]}}

    if slot.state == "text":
        state: Any = topic.short if slot.target is not None and slot.target < 100 else rng.pick(topic.texts)
    elif slot.state == "unicode":
        state = topic.unicode_text
    elif slot.state == "object":
        state = json_state(topic, rng, mixed=slot.name.endswith("mixed") or slot.kind == "multi")
    elif slot.state == "array":
        state = json_array_state(topic, rng)
    elif slot.state == "null":
        state = None
    elif slot.state == "empty":
        state = ""
    else:
        raise ValueError(slot.state)

    if slot.kind == "choice":
        questions = {"answer": choice_question(topic, slot, rng)}
    elif slot.kind == "noul":
        questions = {"answer": noul_question(topic, slot.criteria, rng)}
    elif slot.kind == "score":
        questions = {"answer": score_question(topic, slot.size)}
    elif slot.kind == "multi" and slot.state == "text":
        questions = {
            "route": choice_question(topic, Slot(0, "", "choice", 3), rng),
            "today": noul_question(topic, "both", rng),
            "level": score_question(topic, 3),
        }
    elif slot.kind == "multi":
        questions = {
            "flag": noul_question(topic, "none", rng),
            "route": choice_question(topic, Slot(0, "", "choice", 4), rng),
            "level": score_question(topic, 4),
        }
    else:
        raise ValueError(slot.kind)
    return {"state": state, "questions": questions}


def request_text(request: Dict[str, Any]) -> str:
    return json.dumps(request, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------------------------------
# The shared-domain rules, executable
# ---------------------------------------------------------------------------------------------------------------------

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
I64 = (-(1 << 63), (1 << 63) - 1)


def check_domain(request: Any, letter_keys: bool = False) -> List[str]:
    """The reasons `request` falls outside the shared domain (empty: inside)."""
    problems: List[str] = []

    def text_ok(where: str, value: Any, may_be_empty: bool = False):
        if not isinstance(value, str):
            problems.append(f"{where}: not a string")
            return
        if not may_be_empty and (not value.strip() or value != value.strip()):
            problems.append(f"{where}: blank or padded")
        if "<" in value or ">" in value:
            problems.append(f"{where}: '<' or '>' (a special-token spelling)")

    def json_ok(where: str, value: Any, depth: int = 0):
        if depth > 32:
            problems.append(f"{where}: too deep")
        elif value is None or isinstance(value, bool):
            return
        elif isinstance(value, int):
            if not I64[0] <= value <= I64[1]:
                problems.append(f"{where}: integer outside signed i64")
        elif isinstance(value, float):
            if value != 0.0 and not (1e-4 <= abs(value) < 1e15):
                problems.append(f"{where}: float outside [1e-4, 1e15)")
            elif value != 0.0 and float(f"{value:.6g}") != value:
                problems.append(f"{where}: float with more than 6 significant digits")
            elif value == 0.0 and math.copysign(1.0, value) < 0:
                problems.append(f"{where}: negative zero")
        elif isinstance(value, str):
            text_ok(where, value, may_be_empty=True)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                json_ok(f"{where}[{i}]", item, depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                text_ok(f"{where} key", key)
                json_ok(f"{where}.{key}", item, depth + 1)
        else:
            problems.append(f"{where}: not JSON")

    if not isinstance(request, dict) or set(request) != {"state", "questions"}:
        return ["the request must have exactly `state` and `questions`"]
    state = request["state"]
    if isinstance(state, str):
        text_ok("state", state, may_be_empty=True)
    elif state is not None:
        if not isinstance(state, (dict, list)):
            problems.append("state: not a string, object, array or null")
        json_ok("state", state)
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        return problems + ["questions: not a non-empty object"]
    for qid, q in questions.items():
        where = f"question {qid}"
        if not ID_RE.match(qid):
            problems.append(f"{where}: id")
        if not isinstance(q, dict):
            problems.append(f"{where}: not an object")
            continue
        allowed = {"type", "instructions", "criteria"}
        if set(q) - allowed:
            problems.append(f"{where}: fields beyond type, instructions, criteria")
        text_ok(f"{where} instructions", q.get("instructions"))
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                problems.append(f"{where}: choice criteria")
                continue
            single = all(len(k) == 1 and k.isascii() and k.isalpha() for k in criteria)
            for key, desc in criteria.items():
                if not (ID_RE.match(key) or (letter_keys and single)):
                    problems.append(f"{where}: key {key!r}")
                text_ok(f"{where} option {key}", desc)
        elif kind == "noul":
            if criteria is None and "criteria" not in q:
                continue
            if not isinstance(criteria, dict) or not criteria or set(criteria) - {"true", "false"}:
                problems.append(f"{where}: noul criteria")
                continue
            for key, desc in criteria.items():
                text_ok(f"{where} {key}", desc)
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                problems.append(f"{where}: score levels")
                continue
            for i, level in enumerate(criteria):
                text_ok(f"{where} level {i}", level)
        else:
            problems.append(f"{where}: type {kind!r}")
    return problems


def strict_load(text: str) -> Any:
    """json.loads refusing duplicate keys anywhere."""

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    return json.loads(text, object_pairs_hook=pairs)


# ---------------------------------------------------------------------------------------------------------------------
# Measuring and padding
# ---------------------------------------------------------------------------------------------------------------------

Measure = Callable[[List[str]], List[List[int]]]


class OracleMeasure:
    """Prompt lengths from upstream's own preparation: `decision-oracle run --vocab-only`, nothing evaluated."""

    def __init__(self, binary: Path, vocab: Path):
        self.binary = binary
        self.vocab = vocab
        self.calls = 0

    def __call__(self, texts: List[str]) -> List[List[int]]:
        self.calls += 1
        with tempfile.TemporaryDirectory(prefix="make-heldout-") as work:
            inp, out = Path(work) / "in.json", Path(work) / "out.json"
            cases = [{"id": f"m{i}", "request_text": t} for i, t in enumerate(texts)]
            inp.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
            cmd = [str(self.binary), "run", "--input", str(inp), "--output", str(out), "--vocab-only",
                   str(self.vocab)]
            done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if done.returncode != 0:
                raise RecipeError(f"decision-oracle failed while measuring: {done.stdout.decode()[-2000:]}")
            result = json.loads(out.read_text(encoding="utf-8"))
        lengths = []
        for case in result.get("cases", []):
            tasks = (case.get("prepare") or {}).get("tasks")
            if not tasks:
                raise RecipeError(f"upstream did not prepare a candidate: {json.dumps(case)[:500]}")
            lengths.append([len(t["token_ids"]) for t in tasks])
        if len(lengths) != len(texts):
            raise RecipeError("the oracle returned a different number of cases")
        return lengths


class RecipeError(Exception):
    """The recipe cannot be followed: fix and re-review it; never retry with another seed."""


def filler_sentences(topic: Topic, n: int) -> List[str]:
    """The slot's padding sentences, in order: the topic's filler, cycled, each with its running number."""
    return [f"({i + 1}) {topic.filler[i % len(topic.filler)]}" for i in range(n)]


def with_padding(request: Dict[str, Any], sentences: List[str], words: List[str]) -> Dict[str, Any]:
    """`request` with the filler appended to its state: to a text state, or as a `notes` list of an object state.

    Words go at the end of a text state, and before the final period of an object state's last note, so that the
    note's closing `."` (which tokenizes with the quote) stays as it was and a word adds its own tokens only."""
    padded = json.loads(json.dumps(request))
    if isinstance(padded["state"], str):
        tail = " ".join(sentences + words)
        padded["state"] = (padded["state"] + " " + tail).strip() if tail else padded["state"]
    elif isinstance(padded["state"], dict):
        notes = list(sentences)
        if words:
            last = notes.pop() if notes else ""
            stem, period = (last[:-1], ".") if last.endswith(".") else (last, "")
            notes.append((stem + " " if stem else "") + " ".join(words) + period)
        if notes:
            padded["state"]["notes"] = notes
    else:
        raise RecipeError("only text and object states are padded")
    return padded


# The most word levels one slot's search may take (each level is one batched measurement).
MAX_WORD_LEVELS = 40


def pad(slot: Slot, topic: Topic, request: Dict[str, Any], measure: Measure) -> Tuple[Dict[str, Any], int]:
    """The request padded to exactly `slot.target` prompt tokens (its first pass).

    Sentences first: the most of the slot's filler sentences (in order) that keep the prompt within the target, found
    with two batched measurements (every `step`-th prefix, then each prefix of the last step). Then words, a
    breadth-first search: each level appends one more FILLER_WORDS entry (a later one than the sequence's last) to
    every sequence kept so far, measures all of them in one batch, and keeps the first sequence found for each new
    prompt length within the target; the first sequence to reach the target exactly wins. It starts from the most
    sentences, then one and two fewer. The order is fixed, so the result is too. Lengths must not decrease as text is
    appended. No exact hit fails the recipe."""
    target = slot.target
    assert target is not None
    base = measure([request_text(request)])[0][0]
    if base > target:
        raise RecipeError(f"{case_id(slot)}: the unpadded prompt is {base} tokens, over the target {target}")
    ceiling = max(2, (target - base) // 4 + 8)
    pool = filler_sentences(topic, ceiling)

    def lengths(ks: List[int]) -> Dict[int, int]:
        rows = measure([request_text(with_padding(request, pool[:k], [])) for k in ks])
        found = dict(zip(ks, (row[0] for row in rows)))
        if any(found[a] > found[b] for a, b in zip(ks, ks[1:])):
            raise RecipeError(f"{case_id(slot)}: the prompt got shorter as filler was appended")
        return found

    step = max(1, ceiling // 64)
    coarse_ks = sorted(set(range(0, ceiling + 1, step)) | {ceiling})
    coarse = lengths(coarse_ks)
    if coarse[ceiling] <= target:
        raise RecipeError(f"{case_id(slot)}: {ceiling} filler sentences do not exceed {target} tokens")
    k0 = max(k for k in coarse_ks if coarse[k] <= target)
    fine_ks = sorted(set(range(max(0, k0 - 2), min(k0 + step, ceiling) + 1)))
    fine = lengths(fine_ks)
    n_max = max(k for k in fine_ks if fine[k] <= target)

    def search(sentences: List[str], start: int) -> Optional[List[str]]:
        kept: Dict[int, List[int]] = {start: []}  # prompt length -> the first word-index sequence reaching it
        frontier: List[Tuple[int, List[int]]] = [(start, [])]
        for _ in range(MAX_WORD_LEVELS):
            if target in kept:
                return [FILLER_WORDS[i] for i in kept[target]]
            candidates = [(length, seq + [i]) for length, seq in frontier
                          for i in range((seq[-1] + 1) if seq else 0, len(FILLER_WORDS))]
            if not candidates:
                return None
            rows = measure([request_text(with_padding(request, sentences, [FILLER_WORDS[i] for i in seq]))
                            for _, seq in candidates])
            frontier = []
            for (parent, seq), row in zip(candidates, rows):
                length = row[0]
                if length < parent:
                    raise RecipeError(f"{case_id(slot)}: the prompt got shorter as a word was appended")
                if length <= target and length not in kept:
                    kept[length] = seq
                    frontier.append((length, seq))
        return [FILLER_WORDS[i] for i in kept[target]] if target in kept else None

    for n in range(n_max, max(-1, n_max - 3), -1):
        if n not in fine:
            continue
        words = search(pool[:n], fine[n])
        if words is not None:
            padded = with_padding(request, pool[:n], words)
            length = measure([request_text(padded)])[0][0]
            if length != target:
                raise RecipeError(f"{case_id(slot)}: the padded prompt measures {length}, not {target}")
            return padded, length
    raise RecipeError(f"{case_id(slot)}: no exact padding to {target} tokens")


# ---------------------------------------------------------------------------------------------------------------------
# The whole set
# ---------------------------------------------------------------------------------------------------------------------


def recipe_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def assign_topics(rng: SplitMix64) -> Dict[int, Topic]:
    """A topic per slot: fixed scenarios keep theirs, unicode slots draw from the three with a non-ASCII text, the
    rest cycle through a shuffled order of all twelve."""
    order = rng.shuffled(TOPICS)
    out: Dict[int, Topic] = {}
    i = 0
    for slot in SLOTS:
        if slot.fixed == "ambiguous":
            out[slot.number] = TOPIC_BY_NAME["clinic"]
        elif slot.fixed == "borderline":
            out[slot.number] = TOPIC_BY_NAME["it_incidents"]
        elif slot.state == "unicode":
            out[slot.number] = TOPIC_BY_NAME[rng.pick(UNICODE_TOPICS)]
        else:
            out[slot.number] = order[i % len(order)]
            i += 1
    return out


def generate(measure: Measure, seed: int = SEED) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The 40 cases and their generation record. Only SEED makes the held-out set; tests use other seeds so that no
    held-out request exists before the bounds are frozen."""
    if len(SLOTS) != 40 or [s.number for s in SLOTS] != list(range(1, 41)):
        raise RecipeError("the recipe has exactly 40 slots, numbered in order")
    rng = SplitMix64(seed)
    topics = assign_topics(rng)
    cases, records = [], []
    for slot in SLOTS:
        topic = topics[slot.number]
        request = build_request(slot, topic, rng)
        measured: Optional[int] = None
        if slot.target is not None:
            request, measured = pad(slot, topic, request, measure)
        problems = check_domain(request, letter_keys=slot.keys in ("upper", "lower"))
        if problems:
            raise RecipeError(f"{case_id(slot)}: outside the shared domain: {problems}")
        text = request_text(request)
        if strict_load(text) != request:
            raise RecipeError(f"{case_id(slot)}: the request does not round-trip")
        cases.append({
            "schema_version": 1,
            "id": case_id(slot),
            "profile": PROFILE,
            "title": f"G1 held-out slot {slot.number}: {slot.name.replace('-', ' ')} ({topic.name})",
            "covers": slot.covers,
            "notes": "Written by tools/conformance/decision_g1/make_heldout.py from its fixed recipe; upstream is "
                     "left for the upstream oracle, and the xybrid expectations are proposals.",
            "artifact": ARTIFACT,
            "settings": SETTINGS,
            "request_text": text,
            "upstream": {},
            "xybrid": {"parse": "match", "prepare": "match", "evaluate": "reference", "finalize": "match"},
        })
        records.append({"id": case_id(slot), "topic": topic.name, "questions": list(request["questions"]),
                        "request_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "target_tokens": slot.target, "measured_tokens": measured})
    lengths = measure([c["request_text"] for c in cases])
    for record, passes in zip(records, lengths):
        record["pass_tokens"] = passes
        if record["target_tokens"] is not None and passes[0] != record["target_tokens"]:
            raise RecipeError(f"{record['id']}: measured {passes[0]} tokens, the target is {record['target_tokens']}")
    generation = {
        "schema_version": 1,
        "what": "How G1's held-out cases were generated. No numeric outcome was read.",
        "recipe": {"file": "tools/conformance/decision_g1/make_heldout.py", "sha256": recipe_digest(), "seed": seed},
        "cases": records,
    }
    return cases, generation


def allowlist_for(cases: List[Dict[str, Any]], generation: Dict[str, Any]) -> Dict[str, Any]:
    included = {record["id"]: record["questions"] for record in generation["cases"]}
    return {
        "schema_version": 1,
        "what": "G1's held-out corpus: every generated case is in the shared domain by construction (make_heldout.py "
                "check_domain); the comparator still checks every identity and fails on any mismatch.",
        "corpora": {
            "heldout": {
                "cases": "integration-tests/fixtures/decision/g1/heldout/cases",
                "hf": "integration-tests/fixtures/decision/g1/heldout/hf",
                "oracle": "integration-tests/fixtures/decision/g1/heldout/oracle",
                "counts": {
                    "cases": len(cases),
                    "included_cases": len(included),
                    "included_questions": sum(len(q) for q in included.values()),
                    "excluded_cases": 0,
                },
                "included": included,
                "excluded": {},
            }
        },
    }


def dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oracle", help="decision-oracle (tools/conformance/decision/build.sh)")
    parser.add_argument("--vocab", help="the pinned d1 GGUF whose vocabulary measures prompt lengths")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: %(default)s)")
    parser.add_argument("--check-domain", metavar="DIR", help="check every case file in DIR against the domain rules")
    parser.add_argument("--recipe", action="store_true", help="print the slot table and the recipe digest")
    args = parser.parse_args(argv)

    if args.recipe:
        print(f"recipe sha256 {recipe_digest()}, seed {SEED:#x}")
        for slot in SLOTS:
            print(f"{case_id(slot):40s} {slot.kind:6s} size={slot.size:<3d} keys={slot.keys:9s} state={slot.state:7s} "
                  f"criteria={slot.criteria:4s} target={slot.target} fixed={slot.fixed}")
        return 0
    if args.check_domain:
        failures = 0
        for path in sorted(Path(args.check_domain).glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            try:
                request = strict_load(case["request_text"])
            except ValueError as error:
                print(f"{path.name}: {error}")
                failures += 1
                continue
            letter = any(len(k) == 1 for q in request.get("questions", {}).values()
                         if isinstance(q, dict) and isinstance(q.get("criteria"), dict) for k in q["criteria"])
            for problem in check_domain(request, letter_keys=letter):
                print(f"{path.name}: {problem}")
                failures += 1
        print("domain:", "every case is inside" if not failures else f"{failures} problems")
        return 1 if failures else 0
    if not (args.oracle and args.vocab):
        parser.error("--oracle and --vocab are required to generate")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"make_heldout.py: {out} is not empty; a held-out set is generated once", file=sys.stderr)
        return 1
    measure = OracleMeasure(Path(args.oracle), Path(args.vocab))
    try:
        cases, generation = generate(measure)
    except RecipeError as error:
        print(f"make_heldout.py: the recipe failed: {error}", file=sys.stderr)
        return 1
    generation["measured_with"] = {
        "tool": str(Path(args.oracle).resolve()),
        "tool_sha256": hashlib.sha256(Path(args.oracle).read_bytes()).hexdigest(),
        "vocab_sha256": hashlib.sha256(Path(args.vocab).read_bytes()).hexdigest(),
        "calls": measure.calls,
    }
    (out / "cases").mkdir(parents=True)
    for case in cases:
        (out / "cases" / f"{case['id']}.json").write_text(dump(case), encoding="utf-8")
    (out / "generation.json").write_text(dump(generation), encoding="utf-8")
    (out / "allowlist.json").write_text(dump(allowlist_for(cases, generation)), encoding="utf-8")
    print(f"wrote {len(cases)} cases ({sum(len(r['questions']) for r in generation['cases'])} questions) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
