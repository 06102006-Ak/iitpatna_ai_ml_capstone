SELECT title, price_gbp, price_inr, rating FROM books WHERE in_stock = 1 AND price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp DESC;
