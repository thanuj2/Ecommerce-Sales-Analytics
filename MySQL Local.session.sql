SELECT Country,
SUM(Quantity * UnitPrice) AS total_revenue
FROM ecommerce_sales
GROUP BY Country
ORDER BY total_revenue DESC;