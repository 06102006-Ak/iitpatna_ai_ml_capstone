WITH ranked AS (
    SELECT b.book_id, b.title, b.rating, b.price_inr, c.category_name,
           ROW_NUMBER() OVER (PARTITION BY c.category_name ORDER BY b.rating DESC, b.price_inr DESC, b.title) AS rank_in_category
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
)
SELECT category_name, title, rating, price_inr, rank_in_category
FROM ranked
WHERE rank_in_category <= 3
ORDER BY category_name, rank_in_category;
