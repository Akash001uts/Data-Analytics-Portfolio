# Learnings

A running log of things I've learnt, got wrong, or got caught out by while building this. Newest at the bottom.

## Looking back at v1

Before rebuilding anything, I went back through my original three projects and wrote down what was wrong with them.
It shaped most of the decisions below.

- **My LSTM's multi-day forecast didn't do what I thought.** Each step was meant to feed the last prediction back in
  and slide the window forward. Instead it reset to the same starting window every time and overwrote the last value
  rather than shifting it. The chart I'd labelled "errors compound over time" was really showing that bug.
- **I leaked test data without noticing.** In one model I scaled the whole price series before splitting it into train
  and test, so the scaler had already seen the test prices. It's a small leak, but it's exactly the kind that makes
  results look better than they are.
- **I never compared against a baseline.** For stock prices, "tomorrow's price equals today's" is a surprisingly hard
  baseline to beat, and I never checked whether my model beat it. For sentiment, I said RoBERTa "clearly outperforms"
  VADER without calculating a single accuracy or F1 score.
- **My projects followed tutorials too closely.** The code was mine, but the questions and the structure weren't. This
  time I'm starting from a question I actually care about.

**What I'm doing differently:** every model gets compared with something simpler, every claim gets a number, and the
rules that stop leakage are written as tests so I can't quietly break them.

## Picking where the data comes from (6 Oct 2026)

This turned out to be the most important decision so far.

- **More detail isn't automatically better.** The PHIDU Social Health Atlas has hospital admission rates for about
  1,165 small areas, which looked perfect. Then I noticed it only counts *public* hospitals. Comparing it with the
  AIHW's all-hospital figures, the gap between them lined up strongly with private health insurance. If I'd modelled
  the public-only numbers, the "unexpectedly low admissions" areas would mostly have been wealthy areas where people go
  to private hospitals. I switched to the AIHW's coarser SA3 data, which counts both.
- **Check the year.** The detailed data was only for 2020-21, the middle of COVID, when admissions dropped across the
  board. A one-off year like that would make a strange baseline.
- **Don't assume the data you planned on exists.** My plan included "GP use" as an input, assuming the Social Health
  Atlas had it. It doesn't, at any level. The AIHW's Medicare tables do, but only by SA3. That pushed me further
  towards SA3.
- **Missing data isn't random.** The areas without a published rate include a few very remote ones, which probably
  have some of the highest rates in the country. Dropping them is unavoidable, but it means my model is weakest where
  it matters most, and I need to say that in the write-up instead of hiding it.
- **Read the licence.** PHIDU's data is licensed CC BY-NC-SA. That's fine for a student project, but it means anything
  I make from it has to carry the same licence. I hadn't thought about licences for outputs before.

## Setting the project up properly (7 Oct 2026)

- **Pin your data, because it changes.** I recorded each file's size and SHA256 hash on day one. The next day,
  re-downloading the main PHIDU spreadsheet gave a different file: a new September release had come out (the day
  before, the server had given me an older June copy). The fetch script stopped with an "upstream changed" error instead of
  quietly using the new file. I compared the two versions, re-ran every check from day one on the new one (none of the
  results changed), and then updated the manifest. Without the hashes I wouldn't even have known.
- **The new release renamed a sheet.** `Population_proportion` became `Indigenous_proportion`. That would have broken
  my code. Now a test looks up every input by its sheet, heading and label
  in the real spreadsheet, and fails if any of them can't be found.
- **APIs can answer in a different format depending on how you ask.** The MyHospitals API sent me JSON when I tested it
  with `curl`, but CSV when my script asked, because my test had sent an `Accept: application/json` header and my
  script didn't. Its JSON also starts with a byte-order mark, which Python's JSON reader rejects unless you open the
  file as `utf-8-sig`.
- **Check the rows, not just the column headings.** I'd tested that every column I needed existed. When I added a test
  that the rows were what I expected too, it failed on every sheet. Each state has an extra pseudo-area coded `x9999`
  ("ABS cell adjustment" or "Unknown NSW", for example) sitting in among the real areas. My earlier analysis happened to
  avoid them, but a straightforward parser would have treated them as real places.
- **Write the leakage rules as code.** The list of model inputs is an explicit allowlist in
  `src/dap/health/features.py`. Tests fail if any input comes from a hospital admissions or emergency department sheet,
  from the same source as the thing I'm predicting, or from a count column instead of a rate. It might look like
  overkill, but it means I can't accidentally give the model the answer.
- **Make it runnable by someone else.** I cloned a fresh copy of the repo with no data in it and ran the install,
  lint and tests from scratch, the same way CI does. The tests run on small made-up fixtures, and the ones that need
  the real spreadsheets skip cleanly until you download them. Doing this also caught a fixture file written with
  Windows line endings, and that my pre-commit hooks couldn't find `uv` because of how I'd installed it.

## Cleaning, joining and a first look (7 Oct 2026)

- **Spreadsheet headers aren't consistent, even within one workbook.** In the Census condition sheets, the actual
  condition names ("People who reported they had asthma") sit in the second header row, under a caveat that spans the
  whole sheet. My first fix was to always prefer the second row, which immediately broke a different sheet where the
  second row is a sub-heading inside a block (Housing stress: mortgage, rental, and so on). The workbook tests caught
  it straight away. The reader now keeps both rows and only accepts a match when exactly one column fits.
- **Sometimes it's the test that's wrong.** A check that every remoteness share was between 0 and 1 failed. It turned
  out two near-empty SA3s have no population, so their shares are missing, and a missing value isn't "between 0 and
  1". I made the test say what I actually meant (no missing values where there's a target, and every value that is
  there sits between 0 and 1) instead of patching the data to make the test pass.
- **The obvious input isn't always the useful one.** I assumed GP use would be one of the strongest predictors,
  since PPH is meant to measure primary care. It barely relates to the rate. My first explanation was an age effect,
  because the GP figures are crude rates and the admission rate is age-standardised, but when I checked, it stayed
  near zero after allowing for age. So I've written "I don't know why yet" instead of a neat story I hadn't tested.
- **Check your claims against your own output.** Before pushing, I went back through the notebook text and found
  three sentences that didn't match the numbers above them: a "twice as high" that was really about 1.4 times, a
  "the range widens" that didn't hold for every group, and a pre-COVID comparison that included the ACT even though
  my own notes said I'd left it out. Writing the words before looking closely at the numbers is an easy trap.
- **Look at the outliers by name.** One dot on the scatter plot was very disadvantaged but had a low rate. Looking it
  up gave Fairfield in Sydney, with a large migrant population. A summary statistic would have hidden it.
- **Say what your quintiles are.** I split areas into population-weighted quintiles, so each holds about a fifth of
  the people rather than a fifth of the areas. Both are reasonable, but they give different answers, so the chart has
  to say which one it uses.

## Spatial statistics and models (7 Oct 2026)

- **Test the way the model will be used.** Neighbouring areas have very similar rates (Moran's I is high), so with a
  random split most test areas have a neighbour in the training data. Holding out whole SA4 regions instead knocked
  a noticeable amount off the score of every model that learns from the data, and the most off the most flexible
  one. Reporting only the random
  split would have overstated my results.
- **The best-looking model was the one cheating the most.** The spatial lag model explained most of the
  variation when fitted on every area, and left almost no pattern in its residuals. But its strength was using the
  neighbours' actual rates. To predict an area whose whole region is held out, I had to use the version that only
  needs features, and then it was one of the weaker models on the log scale. Giving it the neighbours' real rates in cross-validation
  would have leaked the answer without any error message.
- **I checked my leakage tests could actually fail.** I wrote a test that changes the test areas' rates and checks the
  predictions don't move. A passing test only means something if it would catch a leak, so I wrote deliberately
  leaky versions (a spatial lag that used neighbours' rates, a mean taken over all rows, and a scaler fitted on
  everything) and checked that each one failed.
- **A weird score is worth chasing.** Ridge regression had a huge spread between repeats on the rate scale but was
  the best model on the log scale. That looked like a bug. It turned out to be extrapolation: when a remote NT region
  is held out, ridge carries its straight line past anything it has seen, and converting back from the log turns a
  moderate miss into a rate several times too high. Trees have the opposite problem and can't predict above their
  training range. Both are real, so I report both scales.
- **Rule out your own mistakes before calling something a finding.** Maryborough in Queensland came out about double
  its expected rate. There's also a Maryborough in Victoria, so my first thought was a bad join. The raw AIHW tables
  show it has been well above its neighbours in every year since 2017-18, so it's real.
- **Generate the numbers, then test that they match.** The results table in the project README is written by
  `dap health train` from `results.json`, and a test fails if the two drift apart. I still checked the hand-written
  notebook text against the outputs, and found two sentences that said more than the numbers did: one claimed both
  models agreed on features they didn't, and one said GP use didn't matter when after-hours GP visits did show up.

## Adding a state term (7 Oct 2026)

- **A random effect doesn't need a mixed-model library.** I wanted each state to get its own intercept, shrunk
  towards zero when a state has little data. statsmodels' mixed models don't take case weights, and every other
  model here is weighted by population, so I wrote the shrinkage myself: each state's mean residual times
  n / (n + k), with k estimated from how much the states differ compared with the noise inside them. It's one short
  class and it wraps any of my models.
- **Training residuals lie about what's left over.** My first idea was to fit LightGBM and then average its
  residuals by state. But LightGBM fits its training areas closely, so those residuals are too small and the state
  offsets would come out too small as well. I get the residuals from a cross-validation inside the training areas
  instead (grouped by SA4 again), and a test checks they're bigger than the in-sample ones. I checked the tests
  could fail too: a version using in-sample residuals and a version taking offsets from every area both got caught.
- **Spatial CV decides which group effects can work at all.** An SA4 random effect would be useless here, because
  the held-out SA4 is never in training. Even state only half works: the ACT is one SA4, so it's never seen when
  it's tested, and the NT has two. The term evened out the big states, but the two territories still sit above
  expected.
- **The same idea can help one model and hurt another.** The state intercept was a clear win for LightGBM but made
  ridge worse on the rate scale, because it stacks on top of ridge's extrapolation in the NT. I only spotted that
  because I report both scales.

## Making the interactive map (7 Oct 2026)

- **Map files get big fast.** My first version of the page was 3.3 MB, almost all of it coastline: the Kimberley
  SA3 alone has about 1,500 islands. Simplifying in degrees would have been the quick fix, but a hundredth of a
  degree is about a kilometre, which would wreck inner-city areas only a few kilometres across. I simplified in
  metres instead, with more detail for the capital cities than the regions, and dropped islands under 2 km². That
  got it to 1.2 MB without changing how it looks.
- **Count what you colour.** A test that the legend counts add up to the number of areas failed by one. Matplotlib's
  `BoundaryNorm` puts the single highest value past the last class. It still got the right colour, so the map
  looked fine, but that area was missing from the legend count.
- **Share the settings between the static and interactive maps.** The page imports its colour classes from the
  same module as the PNG figures, so the two can't quietly disagree.

## Redoing the sentiment project (7 Oct 2026)

- **Almost a third of the dataset was repeats.** Amazon shows a review on every variant of a product, so the same
  text appears many times. I removed duplicates before splitting, and split by reviewer, so no review and no
  reviewer is on both sides. My first version just took the first 500 rows, without removing any.
- **My old claim was right, but I hadn't earned it.** Measured properly, RoBERTa does beat VADER. But a TF-IDF model
  with logistic regression, trained on these reviews, beats RoBERTa, which was trained on tweets and used as is. The paired
  bootstrap gives an interval for each difference, which is the thing I should have reported the first time.
- **Accuracy can go down when a model gets better.** Tuning VADER's thresholds lowered its accuracy and raised its
  macro-F1, because with nearly four in five reviews positive, saying "positive" a lot is an easy way to be
  "accurate".
- **Pin the model, then check you actually got it.** I pinned the RoBERTa model to one commit. The logs showed
  transformers fetching files from an open pull request on the model's page, so I read the library source. It
  loads the pinned weights, and the pull request download is a background "convert it for next time" step, which I
  switched off. I also checked the load report: no weights were missing, so the classifier wasn't randomly
  initialised.
- **My stars check was broken.** My check for reviews that mention "stars" matched nothing, because a
  `\b` in my regex got turned into a backspace character on its way through the shell. The slice test still
  passed, because none of its example texts mentioned stars. I fixed the regex and added a test text that does.
- **Read the mistakes.** Some one-star reviews praise the food and complain about the price, and one reads like a
  five-star review with the wrong rating. No model can get that kind right, so a perfect score isn't possible.

## Tuning RoBERTa's decision rule (7 Oct 2026)

- **Give every model the same chance.** I'd tuned VADER's thresholds on validation but used RoBERTa's plain
  argmax, which made the comparison a bit lopsided. So I scored 5,000 validation reviews with RoBERTa and tuned an
  offset on the log probability of negative and neutral, the same kind of grid search I did for VADER.
- **The fix wasn't where I expected.** I assumed RoBERTa needed a push towards neutral. The best rule actually made
  it slower to say negative, because zero-shot it was calling a lot of three-star reviews negative. Looking at the
  confusion matrix before guessing would have told me that.
- **A real gain can still be a small one.** The paired bootstrap puts the improvement clear of zero, but it closes
  only a small part of the gap to TF-IDF. So the gap is mostly about what RoBERTa learnt, not where it draws its
  lines.
- **Test that tuning can't see the test set.** The test scrambles the test reviewers' labels and RoBERTa outputs and
  checks the tuned rule doesn't move. It also checks that tuning on those scrambled rows would give a different
  rule, so the test would actually catch a leak.

## Tidying up the write-up (7 Oct 2026)

- **Read your outputs like a stranger would.** Going back over everything as one piece, I found the interactive
  map's footer said its licence was CC BY-NC-SA 4.0, while the READMEs and data notes all said 3.0 AU, which is
  PHIDU's own licence. Nothing tested the map's footer text, so the two had disagreed since I built it. Now a
  test checks it against the licence in the data manifest.
- **Write it up as what you did, not what you're going to do.** A lot of my README text was still in the future
  tense from when I was planning ("I'll model the log of the rate", a checklist of steps). Once the work's done,
  that reads like it isn't.
