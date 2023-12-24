declare p string;
SET p = """You are an analytics professional who analyses customer verbatim feedback from client satisfaction survey data. You are required to extract categories and the associated subcategories from the verbatims provided as input. You are required to group and consolidate similar categories. Analyze the verbatim feedback and classify its sentiment as either 'positive', 'negative', or 'neutral' with strong focus on the content and tone. Comparisons to earlier experience should be considered too and evaluated carefully. Given a comment with a unique ID and its identified key themes, please analyze and assign an overall sentiment score for the entire comment , ranging from -5 to +5, based on the content and tone present in the comment. The output for each prompt should be formatted  as shown below.

op_cat:op_sub_cat:op_sentiment:op_sentiment_score:op_group_cat

SYSTEM: You are truthful and never lie. Never make up facts and if you are not 100% sure, reply with why you can not answer in a truthful way. If you do not have enough information in the Review to extract categories or subcategories, return "NA". 

For instance, if someone said, 'The delivery was late, and the product was damaged', 
the category might be 'Delivery Issues' with subcategories 'Late Delivery' , 'Damaged Product'

If a customer is talking about cancelling our product or service or saying "can't wait till end of our contract" or something similar, include churn as a subcategory 

We have a product called meter. It prints postage and stamps. Most of the customers have the product and verbatims can be related to the product itself.

For example, 'Billing Issues' could be consolidated with 'Billing', 'Billing and Machine Issues', 'Billing and Marketing Issues', and 'Billing and Website Navigation'

Make sure number of unique group categories is less than 10. 

input: someone used my address for the creation of an account. I asked to remove my email address from all you distribution lists. apparently, after months, I still exist in your system and cannot be removed; Be sure I'll never become a client of Pitney Bowes ...
Customer Service:Not Trustworthy:Negative:-5:Miscellaneous Issues
"""
;

Create or replace table `smithaargolisinternal.pb_demo.pb_verbatims_sentiment_analysis_1` as
with cte1 as (
 SELECT
    * EXCEPT (ml_generate_text_result),ARRAY(SELECT * FROM 
    UNNEST(SPLIT(ltrim(rtrim(replace(replace(string(ml_generate_text_result['predictions'][0]['content']),'\n',','),'.',''))))) )  AS x

      FROM
        ML.GENERATE_TEXT( MODEL `smithaargolisinternal.pb_demo.llm_model`,
      (SELECT
        CONCAT( p,review) as prompt,        *
      FROM
        `smithaargolisinternal.pb_demo.pb_verbatims`),
          STRUCT(0.0 AS temperature,
            1024 AS max_output_tokens,
            1 AS top_p,
            1 AS top_k,
            1 AS candidate_count ))
        ),cte2 as (
         select 
  review,
  LTRIM(RTRIM(SPLIT(a, ':')[SAFE_OFFSET(0)])) AS op_cat,
  LTRIM(RTRIM(SPLIT(a, ':')[SAFE_OFFSET(1)])) AS op_sub_cat,
  LTRIM(RTRIM(SPLIT(a, ':')[SAFE_OFFSET(2)])) AS op_sentiment,
  LTRIM(RTRIM(SPLIT(a, ':')[SAFE_OFFSET(3)])) AS op_sentiment_score,
  LTRIM(RTRIM(SPLIT(a, ':')[SAFE_OFFSET(4)])) AS op_group_cat from   cte1,
    UNNEST(x) a
        )Select review, upper(op_cat) as category, upper(op_sub_cat) as sub_category, op_sentiment as sentiment,op_sentiment_score as sentiment_score, upper(op_group_cat) as group_category from cte2 group by review, category, sub_category, sentiment,sentiment_score, group_category