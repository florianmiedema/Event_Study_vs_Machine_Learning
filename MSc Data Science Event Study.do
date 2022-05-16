
clear

/* ssc install outreg2 */
/* h outreg2 */

/* Setting the Working Directory */
cd "C:\Users\flori\OneDrive\Documenten\Master Data Science & Society\Thesis\Data"

/* Setting the Estimation Window and the Event Window */ 
local EstWindStart = -180
local EstWindEnd = -5
local EventWindStart = 0
local EventWindEnd = 0
local Distance = `EventWindEnd' - `EstWindStart' + 1

/***************************** Return File ************************************/

/* Importing the Return File */
import delimited "return_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Transforming the Return Variable to a Numeric Variable*/
destring return, replace force

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Setting Up the Trading Dates */
bcal create trdates, from(date) maxgap(10) replace
replace date = bofd("trdates", date)
format date %tbtrdates

/* Setting the Panel Variable and Time Variable */
xtset firmid date

/* Saving the Return File */ 
tempfile returnfile
save `returnfile', replace

clear 

/**************************** Market File *************************************/

/* Importing the Market File */
import delimited "market_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Setting Up the Trading Dates */
bcal create trdates, from(date) maxgap(10) replace
replace date = bofd("trdates", date)
format date %tbtrdates

/* Renaming the Market Returns */
rename return mktr

/* Saving the Market File */ 
tempfile marketreturn
save `marketreturn', replace

/* Merging the Return File with the Market File */ 
use `returnfile'
merge m:1 date using `marketreturn'
drop _merge

/* Saving the combined file */ 
tempfile combinedfile
save `combinedfile', replace

clear

/***************************** Event File *************************************/

/* Setting up the Event File */ 
import delimited "event_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Considering the Weekends */ 
gen dow = dow(date)

replace date = date + 1 if dow == 0      
replace date = date + 2 if dow == 6   

drop dow

/* Setting the Event Dates */ 
rename date eventdate
replace eventdate = bofd("trdates", eventdate)
format eventdate %tbtrdates

/* Making an ID for Every Single Event */ 
egen eventid = group(eventdate firmid)

/* Saving the Event File */ 
tempfile eventdates
save `eventdates', replace

/***************************** Computation ************************************/

/* Creating the Tau */ 
expand `Distance'
bys eventid: gen int tau = _n + `EstWindStart' - 1
assert(tau <=`EventWindEnd')

gen date = eventdate + tau
format date %tbtrdates

/* Merging this file with the combined file */ 
merge m:1 firmid date using `combinedfile'
sort eventid tau
drop if _merge != 3
drop _merge

/* Creating the Normal Returns using the Market Model */ 
qui levelsof eventid, local(eid)
gen NR=.
foreach v in `eid'{
	capture qui reg return mktr if tau <= `EstWindEnd' & tau >= `EstWindStart' & eventid == `v'
	qui predict tmp if eventid == `v', xb
	qui replace NR = tmp if eventid == `v'
	drop tmp
}

/* Checking NR's */
preserve
drop if tau <= `EstWindEnd'
sum NR, d
restore

/* Creating the Abnormal Returns */ 
gen AR = return - NR

/* Check if Everything Went Well */ 
sum AR if tau <= `EstWindEnd'
assert(abs(`r(mean)')<0.00001)

/* Temporarely Drop the Estimation Window */ 
preserve
drop if tau <= `EstWindEnd'

/* Computing the CARs */ 
collapse (sum) AR, by(eventid)
rename AR CAR

/* Running the Regression On the CARs */ 
reg CAR, vce(robust)
outreg2 using table1, word dec(4) replace ctitle(Event window, single day) title(Table 3: CAAR for each event window)
/* outreg2 using table1, word dec(4) append ctitle(Event window, 3 days) */
/* outreg2 using table1, word dec(4) append ctitle(Event window, 10 days) */ 

/* outreg2 using table1.doc, dec(2) replace */

sum CAR, d

/* Restoring the Original Data */ 
restore

/********************************** EVENT WINDOW 3 ****************************/

clear

/* ssc install outreg2 */
/* h outreg2 */

/* Setting the Working Directory */
cd "C:\Users\flori\OneDrive\Documenten\Master Data Science & Society\Thesis\Data"

/* Setting the Estimation Window and the Event Window */ 
local EstWindStart = -180
local EstWindEnd = -5
local EventWindStart = 0
local EventWindEnd = 2
local Distance = `EventWindEnd' - `EstWindStart' + 1

/***************************** Return File ************************************/

/* Importing the Return File */
import delimited "return_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Transforming the Return Variable to a Numeric Variable*/
destring return, replace force

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Setting Up the Trading Dates */
bcal create trdates, from(date) maxgap(10) replace
replace date = bofd("trdates", date)
format date %tbtrdates

/* Setting the Panel Variable and Time Variable */
xtset firmid date

/* Saving the Return File */ 
tempfile returnfile
save `returnfile', replace

clear 

/**************************** Market File *************************************/

/* Importing the Market File */
import delimited "market_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Setting Up the Trading Dates */
bcal create trdates, from(date) maxgap(10) replace
replace date = bofd("trdates", date)
format date %tbtrdates

/* Renaming the Market Returns */
rename return mktr

/* Saving the Market File */ 
tempfile marketreturn
save `marketreturn', replace

/* Merging the Return File with the Market File */ 
use `returnfile'
merge m:1 date using `marketreturn'
drop _merge

/* Saving the combined file */ 
tempfile combinedfile
save `combinedfile', replace

clear

/***************************** Event File *************************************/

/* Setting up the Event File */ 
import delimited "event_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Considering the Weekends */ 
gen dow = dow(date)

replace date = date + 1 if dow == 0      
replace date = date + 2 if dow == 6   

drop dow

/* Setting the Event Dates */ 
rename date eventdate
replace eventdate = bofd("trdates", eventdate)
format eventdate %tbtrdates

/* Making an ID for Every Single Event */ 
egen eventid = group(eventdate firmid)

/* Saving the Event File */ 
tempfile eventdates
save `eventdates', replace

/***************************** Computation ************************************/

/* Creating the Tau */ 
expand `Distance'
bys eventid: gen int tau = _n + `EstWindStart' - 1
assert(tau <=`EventWindEnd')

gen date = eventdate + tau
format date %tbtrdates

/* Merging this file with the combined file */ 
merge m:1 firmid date using `combinedfile'
sort eventid tau
drop if _merge != 3
drop _merge

/* Creating the Normal Returns using the Market Model */ 
qui levelsof eventid, local(eid)
gen NR=.
foreach v in `eid'{
	capture qui reg return mktr if tau <= `EstWindEnd' & tau >= `EstWindStart' & eventid == `v'
	qui predict tmp if eventid == `v', xb
	qui replace NR = tmp if eventid == `v'
	drop tmp
}

/* Checking NR's */
preserve
drop if tau <= `EstWindEnd'
sum NR, d
restore

/* Creating the Abnormal Returns */ 
gen AR = return - NR

/* Check if Everything Went Well */ 
sum AR if tau <= `EstWindEnd'
assert(abs(`r(mean)')<0.00001)

/* Temporarely Drop the Estimation Window */ 
preserve
drop if tau <= `EstWindEnd'

/* Computing the CARs */ 
collapse (sum) AR, by(eventid)
rename AR CAR

/* Running the Regression On the CARs */ 
reg CAR, vce(robust)
/* outreg2 using table1, word dec(4) replace ctitle(Event window, single day) title(Table 3: CAAR for each event window)*/ 
outreg2 using table1, word dec(4) append ctitle(Event window, 3 days) 
/* outreg2 using table1, word dec(4) append ctitle(Event window, 10 days) */ 

/* outreg2 using table1.doc, dec(2) replace */

sum CAR, d

/* Restoring the Original Data */ 
restore

/***************************** EVENT WINDOW 10 ********************************/

clear

/* ssc install outreg2 */
/* h outreg2 */

/* Setting the Working Directory */
cd "C:\Users\flori\OneDrive\Documenten\Master Data Science & Society\Thesis\Data"

/* Setting the Estimation Window and the Event Window */ 
local EstWindStart = -180
local EstWindEnd = -5
local EventWindStart = 0
local EventWindEnd = 9
local Distance = `EventWindEnd' - `EstWindStart' + 1

/***************************** Return File ************************************/

/* Importing the Return File */
import delimited "return_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Transforming the Return Variable to a Numeric Variable*/
destring return, replace force

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Setting Up the Trading Dates */
bcal create trdates, from(date) maxgap(10) replace
replace date = bofd("trdates", date)
format date %tbtrdates

/* Setting the Panel Variable and Time Variable */
xtset firmid date

/* Saving the Return File */ 
tempfile returnfile
save `returnfile', replace

clear 

/**************************** Market File *************************************/

/* Importing the Market File */
import delimited "market_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Setting Up the Trading Dates */
bcal create trdates, from(date) maxgap(10) replace
replace date = bofd("trdates", date)
format date %tbtrdates

/* Renaming the Market Returns */
rename return mktr

/* Saving the Market File */ 
tempfile marketreturn
save `marketreturn', replace

/* Merging the Return File with the Market File */ 
use `returnfile'
merge m:1 date using `marketreturn'
drop _merge

/* Saving the combined file */ 
tempfile combinedfile
save `combinedfile', replace

clear

/***************************** Event File *************************************/

/* Setting up the Event File */ 
import delimited "event_file.csv", encoding(Big5)

/* Dropping the Unnecessary Variable */
drop v1

/* Temporarily Renaming the Date Variable */
rename date date2

/* Destringing the Date Variable */
gen date = date(date2, "YMD")
format date %td

/* Dropping the String Variable */ 
drop date2

/* Considering the Weekends */ 
gen dow = dow(date)

replace date = date + 1 if dow == 0      
replace date = date + 2 if dow == 6   

drop dow

/* Setting the Event Dates */ 
rename date eventdate
replace eventdate = bofd("trdates", eventdate)
format eventdate %tbtrdates

/* Making an ID for Every Single Event */ 
egen eventid = group(eventdate firmid)

/* Saving the Event File */ 
tempfile eventdates
save `eventdates', replace

/***************************** Computation ************************************/

/* Creating the Tau */ 
expand `Distance'
bys eventid: gen int tau = _n + `EstWindStart' - 1
assert(tau <=`EventWindEnd')

gen date = eventdate + tau
format date %tbtrdates

/* Merging this file with the combined file */ 
merge m:1 firmid date using `combinedfile'
sort eventid tau
drop if _merge != 3
drop _merge

/* Creating the Normal Returns using the Market Model */ 
qui levelsof eventid, local(eid)
gen NR=.
foreach v in `eid'{
	capture qui reg return mktr if tau <= `EstWindEnd' & tau >= `EstWindStart' & eventid == `v'
	qui predict tmp if eventid == `v', xb
	qui replace NR = tmp if eventid == `v'
	drop tmp
}

/* Checking NR's */
preserve
drop if tau <= `EstWindEnd'
sum NR, d
restore

/* Creating the Abnormal Returns */ 
gen AR = return - NR

/* Check if Everything Went Well */ 
sum AR if tau <= `EstWindEnd'
assert(abs(`r(mean)')<0.00001)

/* Temporarely Drop the Estimation Window */ 
preserve
drop if tau <= `EstWindEnd'

/* Computing the CARs */ 
collapse (sum) AR, by(eventid)
rename AR CAR

/* Running the Regression On the CARs */ 
reg CAR, vce(robust)
/*outreg2 using table1, word dec(4) replace ctitle(Event window, single day) title(Table 3: CAAR for each event window)*/
/* outreg2 using table1, word dec(4) append ctitle(Event window, 3 days) */
outreg2 using table1, word dec(4) append ctitle(Event window, 10 days)

/* outreg2 using table1.doc, dec(2) replace */

sum CAR, d

/* Restoring the Original Data */ 
restore