CREATE TABLE `connections` (
	`owner` text PRIMARY KEY NOT NULL,
	`url` text NOT NULL,
	`encrypted_key` text NOT NULL
);
