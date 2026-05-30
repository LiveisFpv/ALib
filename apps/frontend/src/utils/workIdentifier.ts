const OPENALEX_WORK_RE = /^W\d+$/i
const OPENALEX_URL_RE = /^https?:\/\/(?:www\.|api\.)?openalex\.org\/(?:works\/)?/i
const DOI_RE = /10\.\d{4,9}\/\S+/i

function clean(value: string | undefined | null) {
  const text = String(value ?? '').trim()
  return text || ''
}

export function normalizeOpenAlexWorkId(value: string | undefined | null) {
  const text = clean(value).replace(OPENALEX_URL_RE, '').replace(/^\/+|\/+$/g, '')
  return OPENALEX_WORK_RE.test(text) ? text.toUpperCase() : ''
}

export function normalizeDoi(value: string | undefined | null) {
  let text = clean(value)
  text = text.replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
  text = text.replace(/^doi\s*:?\s*/i, '')
  const match = text.match(DOI_RE)
  return match ? match[0].replace(/[.,;]+$/g, '').toLowerCase() : ''
}

export function normalizeWorkIdentifier(value: string | undefined | null) {
  return normalizeOpenAlexWorkId(value) || normalizeDoi(value) || clean(value)
}

export function normalizeWorkIdentifierList(values: Array<{ id?: string }> | string[] | undefined) {
  const seen = new Set<string>()
  const result: string[] = []
  for (const item of values ?? []) {
    const raw = typeof item === 'string' ? item : item.id
    const normalized = normalizeWorkIdentifier(raw)
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    result.push(normalized)
  }
  return result
}
