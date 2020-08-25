-- Transfer files to docker image 
-- docker cp ./train_without_quotes.csv dev_hana_1:/tmp 
 CONNECT SYSTEM PASSWORD Manager1 
;

ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini',
	 'database') 
SET ('import_export',
	 'enable_csv_import_path_filter') = 'false' WITH RECONFIGURE 
;

 CONNECT DS_USER PASSWORD Password1 
;

 DROP table "DS_USER"."train" 
;

 CREATE TABLE "DS_USER"."train" ("id" INTEGER,
	 "qid1" INTEGER,
	 "qid2" INTEGER,
	 "question1" NVARCHAR(2000),
	 "question2" NVARCHAR(2000),
	 "is_duplicate" SMALLINT ) 
;

 IMPORT 
FROM CSV FILE '/tmp/train_without_quotes_final.csv' 
INTO "DS_USER"."train" WITH RECORD DELIMITED BY '\n' FIELD DELIMITED BY ',' OPTIONALLY ENCLOSED BY '"' SKIP FIRST 1 ROW ERROR LOG '/tmp/err.csv' FAIL ON INVALID DATA 
;
 select
	 count(*) 
from "train"

-- # of unique ids in train
select count(distinct "id"),count(distinct "qid1"),count(distinct "qid2") from "train";

-- # of unique qid1,qid2 in train
select count(distinct "qid1"||"qid2") from "train";

-- # of qid1=qid2
select count(*) from "train" where "qid1"="qid2";

-- of unique {qid1} union {qid2}
select count(*) from (select "qid1" from "train" union select "qid2" from "train");

-- # of question1=question2
select count(*) from "train" where "question1"="question2";

-- # of lower("question1")=lower("question2")
select count(*) from "train" where lower("question1")=lower("question2");

select * from "train" where lower("question1")=lower("question2");

-- # of question1=question2 and !duplicate
select count(*) from "train" where lower("question1")=lower("question2") AND "is_duplicate"=0;

-- of is_duplicate
SELECT COUNT(*) from "train" where "is_duplicate"=1;

-- # of !is_duplicate
SELECT COUNT(*) from "train" where "is_duplicate"=0;

-- # of is_duplicate ! is null
SELECT COUNT(*) from "train" where "is_duplicate" IS NULL;


-- # of is_duplicate <>(0,1)
SELECT COUNT(*) from "train" where "is_duplicate" <>1 and "is_duplicate"<>0;

-- select question1 is null or empty
SELECT COUNT(*) from "train" where "question1" is null or length("question1")=0

-- select question2 is null or empty
SELECT COUNT(*) from "train" where "question2" is null or length("question2")=0


-- (A,B,duplicated) (B,C,duplicated) 
SELECT COUNT(*) from "train" t1, "train" t2 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1;

SELECT t1."qid1" as "t1_qid1",t1."qid2" as "t1_qid2",t2."qid1" as "t2_qid1",t2."qid2" as "t2_qid2",t1."question1" as "t1_question1",t1."question2" as "t1_question2",t2."question2" as "t2_question2" from "train" t1, "train" t2 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1;

SELECT t1."question1" as "t1_question1",t1."question2" as "t1_question2",t2."question2" as "t2_question2" from "train" t1, "train" t2 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1;

-- (A,B,duplicated) (B,C,duplicated) (A,C,?)
SELECT COUNT(*) from "train" t1, "train" t2, "train" t3 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1 and ((t3."qid1"=t1."qid1" and t3."qid2"=t2."qid2") or (t3."qid2"=t1."qid1" and t3."qid1"=t2."qid2"));

SELECT t3.* from "train" t1, "train" t2, "train" t3 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1 and ((t3."qid1"=t1."qid1" and t3."qid2"=t2."qid2") or (t3."qid2"=t1."qid1" and t3."qid1"=t2."qid2")) order by t3."id";


-- (A,B,duplicated) (B,C,duplicated) (A,C,!duplicated)
SELECT COUNT(*) from "train" t1, "train" t2, "train" t3 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1 and ((t3."qid1"=t1."qid1" and t3."qid2"=t2."qid2") or (t3."qid2"=t1."qid1" and t3."qid1"=t2."qid2")) and t3."is_duplicate"=0;

SELECT t3.* from "train" t1, "train" t2, "train" t3 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1 and ((t3."qid1"=t1."qid1" and t3."qid2"=t2."qid2") or (t3."qid2"=t1."qid1" and t3."qid1"=t2."qid2")) and t3."is_duplicate"=0 order by t3."id";

SELECT t1."question1",t1."question2",t2."question1",t2."question2",t3."question1",t3."question2" from "train" t1, "train" t2, "train" t3 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1 and ((t3."qid1"=t1."qid1" and t3."qid2"=t2."qid2") or (t3."qid2"=t1."qid1" and t3."qid1"=t2."qid2")) and t3."is_duplicate"=0 order by t3."id";

SELECT t3."question1",t3."question2" from "train" t1, "train" t2, "train" t3 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1 and ((t3."qid1"=t1."qid1" and t3."qid2"=t2."qid2") or (t3."qid2"=t1."qid1" and t3."qid1"=t2."qid2")) and t3."is_duplicate"=0 order by t3."id";

SELECT t3."question1",t3."question2" from "train" t1, "train" t2, "train" t3 WHERE t1."qid2"=t2."qid1" and t1."is_duplicate"=1 and t2."is_duplicate"=1 and ((t3."qid1"=t1."qid1" and t3."qid2"=t2."qid2") or (t3."qid2"=t1."qid1" and t3."qid1"=t2."qid2")) and t3."is_duplicate"=0 group by t3."question1",t3."question2" ;



-- min, avg, max (len)

select min(length("question1")),max(length("question1")),avg(length("question1")),min(length("question2")),max(length("question2")),avg(length("question2")) from "train";

--  distribution of length

select "length",count(*) as "nb" from (select length("question1") as "length" from "train") group by "length" order by  "length" desc;

-- unique questions
select count(*) from (select "question1" from "train" union select "question2" from "train");

select "question1" as "question" from "train" union select "question2" as "question" from "train";
select count(distinct "question1" ) from "train";
select count(distinct "question2" ) from "train";

--  distribution of length of unique questions 

select "length",count(*) as "nb" from (select length("question") as "length" from (select "question1" as "question" from "train" union select "question2" as "question" from "train") ) group by "length" order by  "length";


-- distribution of length of question1
select "length",count(*) as "nb" from (select length("question1") as "length" from "train" ) group by "length" order by  "length";

-- distribution of length of question2
select "length",count(*) as "nb" from (select length("question2") as "length" from "train" ) group by "length" order by  "length";


drop view "length_uniques";
create view "length_uniques" as select length("question") as "length" from (select "question1" as "question" from "train" union select "question2" as "question" from "train") ;

select "length",count(*) as "nb" from "length_uniques" group by "length" order by "length"


drop view "distrib_length_uniques";
create view "distrib_length_uniques" as select "length",count(*) as "nb" from "length_uniques" group by "length" order by "length"

