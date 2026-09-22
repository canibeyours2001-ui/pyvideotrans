import {sqliteTable,text} from 'drizzle-orm/sqlite-core';
export const connections=sqliteTable('connections',{owner:text('owner').primaryKey(),url:text('url').notNull(),encryptedKey:text('encrypted_key').notNull()});
