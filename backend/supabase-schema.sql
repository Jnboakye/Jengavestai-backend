-- JengaVest — Supabase schema
-- Run this in the Supabase SQL editor (SQL → New query → paste → Run).
-- The backend owns Supabase; row-level security keeps each user's rows private.

-- ── Portfolio holdings ─────────────────────────────────────────────────────
create table if not exists public.holdings (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null references auth.users (id) on delete cascade,
  ticker         text not null,
  name           text not null,
  sector         text not null,
  amount_usd     numeric not null,
  purchase_price numeric not null,
  added_at       timestamptz not null default now(),
  unique (user_id, ticker)
);

alter table public.holdings enable row level security;

drop policy if exists "holdings are private" on public.holdings;
create policy "holdings are private" on public.holdings
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- ── Chat history (conversations + messages) ────────────────────────────────
create table if not exists public.conversations (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users (id) on delete cascade,
  title      text not null default 'New conversation',
  created_at timestamptz not null default now()
);

alter table public.conversations enable row level security;

drop policy if exists "conversations are private" on public.conversations;
create policy "conversations are private" on public.conversations
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create table if not exists public.messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations (id) on delete cascade,
  user_id         uuid not null references auth.users (id) on delete cascade,
  role            text not null check (role in ('user', 'assistant')),
  content         text not null,
  citations       jsonb,
  created_at      timestamptz not null default now()
);

alter table public.messages enable row level security;

drop policy if exists "messages are private" on public.messages;
create policy "messages are private" on public.messages
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index if not exists messages_conversation_idx on public.messages (conversation_id, created_at);
