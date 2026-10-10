-- =====================================================================
-- reports.sql  |  Part 1, Task 3: nine business reports (a-i)
-- Run on the RAW data (duplicates and messy casing still present) after
-- schema.sql and seed_data.sql.
-- Each query is preceded by its exact output as "-- |" comment lines.
-- Regenerate those comment blocks with:  python sql/run_sql.py --refresh-reports
--
-- Revenue per order = quantity * price * (1 - COALESCE(discount_pct, 0) / 100.0)
-- NOTE the 100.0: discount_pct is an INTEGER column, so "/ 100" would be integer
-- division and silently turn every discount into 0.
-- =====================================================================

-- ===== (a) Order totals =====
-- Total orders, total revenue and average order value (NULL discount = 0%).
-- Output:
-- | total_orders | total_revenue | avg_order_value |
-- | 180 | 99860.2 | 554.78 |
SELECT COUNT(*) AS total_orders,
       ROUND(SUM(o.quantity * p.price * (1 - COALESCE(o.discount_pct, 0) / 100.0)), 2) AS total_revenue,
       ROUND(AVG(o.quantity * p.price * (1 - COALESCE(o.discount_pct, 0) / 100.0)), 2) AS avg_order_value
FROM orders o
JOIN products p ON p.product_id = o.product_id;

-- ===== (b) COUNT(*) vs COUNT(column) =====
-- COUNT(*) counts rows; COUNT(rating) skips NULLs. The gap = orders with no rating.
-- Output:
-- | total_rows | orders_with_rating | orders_without_rating |
-- | 180 | 165 | 15 |
SELECT COUNT(*)                AS total_rows,
       COUNT(rating)           AS orders_with_rating,
       COUNT(*) - COUNT(rating) AS orders_without_rating
FROM orders;

-- ===== (c) Customers with zero orders, two ways =====
-- Method 1: LEFT JOIN and HAVING COUNT(order_id) = 0
-- Output:
-- | customer_id | name |
-- | C045 | Vihaan |
SELECT c.customer_id, c.name
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name
HAVING COUNT(o.order_id) = 0
ORDER BY c.customer_id;

-- Method 2: NOT IN with DISTINCT (safe because orders.customer_id is NOT NULL)
-- Output:
-- | customer_id | name |
-- | C045 | Vihaan |
SELECT c.customer_id, c.name
FROM customers c
WHERE c.customer_id NOT IN (SELECT DISTINCT customer_id FROM orders)
ORDER BY c.customer_id;

-- Check: rows that differ between the two methods (must be 0)
-- Output:
-- | mismatching_rows |
-- | 0 |
SELECT (SELECT COUNT(*) FROM (
            SELECT c.customer_id FROM customers c
            LEFT JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_id IS NULL
            EXCEPT
            SELECT customer_id FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)))
     + (SELECT COUNT(*) FROM (
            SELECT customer_id FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)
            EXCEPT
            SELECT c.customer_id FROM customers c
            LEFT JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_id IS NULL))
       AS mismatching_rows;

-- ===== (d) Return rate by city, only cities above 20% =====
-- HAVING filters AFTER aggregation (WHERE cannot see the rate).
-- Output:
-- | city | total_orders | returned_orders | return_rate_pct |
-- | Jaipur | 19 | 8 | 42.1 |
-- | Lucknow | 49 | 15 | 30.6 |
-- | Bangalore | 33 | 8 | 24.2 |
SELECT c.city,
       COUNT(*)        AS total_orders,
       SUM(o.returned) AS returned_orders,
       ROUND(100.0 * SUM(o.returned) / COUNT(*), 1) AS return_rate_pct
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY c.city
HAVING return_rate_pct > 20
ORDER BY return_rate_pct DESC, c.city;

-- ===== (e) Top 5 customers by spend, then ranks 3-5 via OFFSET =====
-- Top 5 (customer_id ASC breaks equal-spend ties deterministically).
-- Output:
-- | customer_id | name | total_spend |
-- | C043 | Reyansh | 12920.0 |
-- | C026 | Isha | 8371.6 |
-- | C008 | Meera | 4564.6 |
-- | C011 | Arjun | 4111.0 |
-- | C042 | Sanya | 3785.0 |
SELECT c.customer_id, c.name,
       ROUND(SUM(o.quantity * p.price * (1 - COALESCE(o.discount_pct, 0) / 100.0)), 2) AS total_spend
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
GROUP BY c.customer_id, c.name
ORDER BY total_spend DESC, c.customer_id
LIMIT 5;

-- Ranks 3-5: skip the first 2 rows, take 3
-- Output:
-- | customer_id | name | total_spend |
-- | C008 | Meera | 4564.6 |
-- | C011 | Arjun | 4111.0 |
-- | C042 | Sanya | 3785.0 |
SELECT c.customer_id, c.name,
       ROUND(SUM(o.quantity * p.price * (1 - COALESCE(o.discount_pct, 0) / 100.0)), 2) AS total_spend
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
GROUP BY c.customer_id, c.name
ORDER BY total_spend DESC, c.customer_id
LIMIT 3 OFFSET 2;

-- ===== (f) Order count and revenue by product category (three-table JOIN) =====
-- orders + products + customers (the customers join confirms every order has a valid customer).
-- Output:
-- | category | order_count | category_revenue |
-- | Haircare | 54 | 44956.1 |
-- | Skincare | 60 | 27346.0 |
-- | Babycare | 30 | 16805.0 |
-- | PersonalCare | 36 | 10753.1 |
SELECT p.category,
       COUNT(*) AS order_count,
       ROUND(SUM(o.quantity * p.price * (1 - COALESCE(o.discount_pct, 0) / 100.0)), 2) AS category_revenue
FROM orders o
JOIN products  p ON p.product_id  = o.product_id
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY p.category
ORDER BY category_revenue DESC;

-- ===== (g) Customers whose name starts with "A" =====
-- LIKE is case-insensitive for ASCII in SQLite; 'A%' = starts with A.
-- Output:
-- | customer_id | name | city |
-- | C001 | Aarav | Mumbai |
-- | C003 | Aditi | Mumbai |
-- | C031 | Aditya | Jaipur |
-- | C036 | Aisha | Delhi |
-- | C004 | Ananya | Lucknow |
-- | C030 | Anika | Bangalore |
-- | C044 | Aria | Bangalore |
-- | C011 | Arjun | Bangalore |
-- | C021 | Aryan | Bangalore |
-- | C041 | Ayaan | Lucknow |
SELECT customer_id, name, city
FROM customers
WHERE name LIKE 'A%'
ORDER BY name;

-- ===== (h) Distinct acquisition sources =====
-- Output:
-- | acquisition_source |
-- | Ad |
-- | Organic |
-- | Referral |
-- | Social |
SELECT DISTINCT acquisition_source
FROM customers
ORDER BY acquisition_source;

-- ===== (i) Add loyalty_tier: Gold for tier-1 cities, else Silver =====
ALTER TABLE customers ADD COLUMN loyalty_tier VARCHAR(10);

UPDATE customers
SET loyalty_tier = CASE WHEN city_tier = 1 THEN 'Gold' ELSE 'Silver' END;

-- Verify the update
-- Output:
-- | loyalty_tier | customers |
-- | Gold | 28 |
-- | Silver | 17 |
SELECT loyalty_tier, COUNT(*) AS customers
FROM customers
GROUP BY loyalty_tier
ORDER BY loyalty_tier;
