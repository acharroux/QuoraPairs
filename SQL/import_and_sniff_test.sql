-- Transfer files to docker image 
-- docker cp ./test_without_quotes.csv dev_hana_1:/tmp 
 CONNECT SYSTEM PASSWORD Manager1 
;

ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini',
	 'database') 
SET ('import_export',
	 'enable_csv_import_path_filter') = 'false' WITH RECONFIGURE 
;

 CONNECT DS_USER PASSWORD Password1 
;

 DROP table "DS_USER"."test" 
;

 CREATE TABLE "DS_USER"."test" ("id" INTEGER,
	 "question1" NVARCHAR(2000),
	 "question2" NVARCHAR(2000)) 
;

 IMPORT 
FROM CSV FILE '/tmp/test_without_quotes_final.csv' 
INTO "DS_USER"."test" WITH RECORD DELIMITED BY '\n' FIELD DELIMITED BY ',' OPTIONALLY ENCLOSED BY '"' SKIP FIRST 1 ROW ERROR LOG '/tmp/err.csv' FAIL ON INVALID DATA 
;
 select
	 count(*) 
from "test"

-- # of unique ids in test
select count(distinct "id") from "test";

-- # of question1=question2
select count(*) from "test" where "question1"="question2";

-- # of lower(question1)=lower(question2)
select count(*) from "test" where lower("question1")=lower("question2");

select * from "test" where lower("question1")=lower("question2");


-- select question1 is null or empty
SELECT COUNT(*) from "test" where "question1" is null or length("question1")=0

-- select question2 is null or empty
SELECT COUNT(*) from "test" where "question2" is null or length("question2")=0


-- min, avg, max (len)

select min(length("question1")),max(length("question1")),avg(length("question1")),min(length("question2")),max(length("question2")),avg(length("question2")) from "test";

--  distribution of length

select "length",count(*) as "nb" from (select length("question1") as "length" from "test") group by "length" order by  "length" desc;

-- unique questions
select count(*) from (select "question1" from "test" union select "question2" from "test");

select "question1" as "question" from "test" union select "question2" as "question" from "test";
select count(distinct "question1" ) from "test";
select count(distinct "question2" ) from "test";

--  distribution of length of unique questions 

select "length",count(*) as "nb" from (select length("question") as "length" from (select "question1" as "question" from "test" union select "question2" as "question" from "test") ) group by "length" order by  "length";


-- distribution of length of question1
select "length",count(*) as "nb" from (select length("question1") as "length" from "test" ) group by "length" order by  "length";

-- distribution of length of question2
select "length",count(*) as "nb" from (select length("question2") as "length" from "test" ) group by "length" order by  "length";


drop view "length_uniques";
create view "length_uniques" as select length("question") as "length" from (select "question1" as "question" from "test" union select "question2" as "question" from "test") ;

select "length",count(*) as "nb" from "length_uniques" group by "length" order by "length"


drop view "distrib_length_uniques";
create view "distrib_length_uniques" as select "length",count(*) as "nb" from "length_uniques" group by "length" order by "length"

