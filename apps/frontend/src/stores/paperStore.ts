import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { AlibApi } from '@/api/useAlibApi'
import { normalizeWorkIdentifier, normalizeWorkIdentifierList } from '@/utils/workIdentifier'
import type {
  PaperResponse,
  SubmissionListQuery,
  SubmissionRecord,
  SubmissionStatus,
  SubmissionUpsertRequest,
} from '@/api/types'

export type PaperStatus = SubmissionStatus

export interface PaperLink {
  id: string
}

export interface PaperSummary {
  id: string
  title: string
  status: PaperStatus
  updatedAt: string
  submittedAt?: string
  moderatorComment?: string
  approvedPaperId?: number
}

export interface PaperDetail extends PaperSummary {
  source: 'submission' | 'catalog'
  source_identifier?: string
  abstract?: string
  year?: number
  best_oa_location?: string
  related_paper: PaperLink[]
  referenced_paper: PaperLink[]
  createdByUserId: number
  moderatedByUserId?: number
  createdAt?: string
  moderatedAt?: string
}

export interface PaperPayload {
  id?: string
  source_identifier?: string
  title?: string
  abstract?: string
  year?: number
  best_oa_location?: string
  related_paper?: PaperLink[]
  referenced_paper?: PaperLink[]
}

const EDITABLE_STATUSES = new Set<PaperStatus>(['draft', 'rejected'])

function normalizeLinkArray(values: string[] | undefined): PaperLink[] {
  return (values ?? []).map((value) => ({ id: value }))
}

function mapSubmission(submission: SubmissionRecord): PaperDetail {
  return {
    id: String(submission.submission_id),
    source: 'submission',
    title: submission.title?.trim() || '',
    status: submission.status,
    updatedAt: submission.updated_at || '',
    submittedAt: submission.submitted_at || undefined,
    moderatorComment: submission.moderation_comment || undefined,
    approvedPaperId: submission.approved_paper_id || undefined,
    source_identifier: submission.source_identifier || undefined,
    abstract: submission.abstract || undefined,
    year: submission.year || undefined,
    best_oa_location: submission.best_oa_location || undefined,
    related_paper: normalizeLinkArray(submission.related_works),
    referenced_paper: normalizeLinkArray(submission.referenced_works),
    createdByUserId: submission.created_by_user_id,
    moderatedByUserId: submission.moderated_by_user_id || undefined,
    createdAt: submission.created_at || undefined,
    moderatedAt: submission.moderated_at || undefined,
  }
}

function mapCatalogPaper(paper: PaperResponse): PaperDetail | null {
  const paperId = String(paper.id || '').trim()
  if (!paperId) return null
  const numericPaperId = Number(paperId)
  return {
    id: paperId,
    source: 'catalog',
    title: paper.title?.trim() || '',
    status: 'approved',
    updatedAt: '',
    approvedPaperId: Number.isFinite(numericPaperId) && numericPaperId > 0 ? numericPaperId : undefined,
    source_identifier: pickPrimaryIdentifier(paper),
    abstract: paper.abstract || undefined,
    year: paper.year || undefined,
    best_oa_location: paper.best_oa_location || undefined,
    related_paper: normalizeLinkArray(paper.related_works),
    referenced_paper: normalizeLinkArray(paper.referenced_works),
    createdByUserId: 0,
  }
}

function pickPrimaryIdentifier(paper: PaperResponse): string | undefined {
  const identifiers = paper.identifiers ?? []
  const preferred = identifiers.find((item) => item.type === 'doi') || identifiers[0]
  return preferred?.value || paper.id || undefined
}

function mergeMyPapers(submissions: PaperDetail[], catalog: PaperDetail[]) {
  const approvedPaperIds = new Set(
    submissions
      .map((paper) => paper.approvedPaperId)
      .filter((value): value is number => typeof value === 'number' && value > 0)
      .map(String),
  )
  const catalogWithoutPublishedSubmissions = catalog.filter((paper) => {
    const paperId = paper.approvedPaperId ? String(paper.approvedPaperId) : paper.id
    return !approvedPaperIds.has(paperId)
  })
  return [...submissions, ...catalogWithoutPublishedSubmissions]
}

function mapSubmissionInput(payload: PaperPayload): SubmissionUpsertRequest {
  return {
    source_identifier: normalizeWorkIdentifier(payload.source_identifier) || '',
    title: payload.title?.trim() || '',
    abstract: payload.abstract?.trim() || '',
    year: payload.year || 0,
    best_oa_location: payload.best_oa_location?.trim() || '',
    related_works: normalizeWorkIdentifierList(payload.related_paper),
    referenced_works: normalizeWorkIdentifierList(payload.referenced_paper),
  }
}

export const usePaperStore = defineStore('paper', () => {
  const items = ref<PaperDetail[]>([])
  const isLoading = ref(false)
  const lastLoaded = ref<string | null>(null)

  const papers = computed(() => items.value)

  const editablePaperIds = computed(() =>
    papers.value
      .filter((paper) => paper.source === 'submission' && EDITABLE_STATUSES.has(paper.status))
      .map((paper) => paper.id),
  )

  function canEdit(id: string, source?: PaperDetail['source']) {
    if (source === 'catalog') return false
    return editablePaperIds.value.includes(id)
  }

  function canDelete(id: string, source?: PaperDetail['source']) {
    return canEdit(id, source)
  }

  function upsertPaper(submission: SubmissionRecord) {
    const next = mapSubmission(submission)
    const index = items.value.findIndex(
      (paper) => paper.source === 'submission' && paper.id === next.id,
    )
    if (index >= 0) {
      items.value.splice(index, 1, next)
    } else {
      items.value.unshift(next)
    }
    return next
  }

  function getMyPapers(statuses?: PaperStatus[]): PaperSummary[] {
    const allowed = statuses?.length ? new Set(statuses) : null
    return papers.value
      .filter((paper) => !allowed || allowed.has(paper.status))
      .map((paper) => ({
        id: paper.id,
        title: paper.title,
        status: paper.status,
        updatedAt: paper.updatedAt,
        submittedAt: paper.submittedAt,
        moderatorComment: paper.moderatorComment,
        approvedPaperId: paper.approvedPaperId,
      }))
  }

  function getById(id: string) {
    return (
      papers.value.find((paper) => paper.source === 'submission' && paper.id === id) ??
      papers.value.find((paper) => paper.id === id)
    )
  }

  async function loadMyPapers(query: SubmissionListQuery = {}) {
    if (isLoading.value) return
    isLoading.value = true
    try {
      const [submissionsResponse, catalogResponse] = await Promise.all([
        AlibApi.listMySubmissions({
          limit: query.limit ?? 100,
          offset: query.offset ?? 0,
          statuses: query.statuses,
        }),
        AlibApi.listAuthorPapers(),
      ])
      const submissions = submissionsResponse.items.map(mapSubmission)
      const catalog = (catalogResponse.papers ?? [])
        .map(mapCatalogPaper)
        .filter((paper): paper is PaperDetail => paper !== null)
      items.value = mergeMyPapers(submissions, catalog)
      lastLoaded.value = new Date().toISOString()
    } finally {
      isLoading.value = false
    }
  }

  async function fetchSubmission(id: string | number) {
    const response = await AlibApi.getMySubmission(id)
    return upsertPaper(response.submission)
  }

  async function saveDraft(payload: PaperPayload) {
    const request = mapSubmissionInput(payload)
    const response = payload.id
      ? await AlibApi.updateMySubmission(payload.id, request)
      : await AlibApi.createSubmission(request)
    return upsertPaper(response.submission)
  }

  async function submitExisting(id: string | number) {
    const response = await AlibApi.submitMySubmission(id)
    return upsertPaper(response.submission)
  }

  async function submitPaper(payload: PaperPayload) {
    const saved = await saveDraft(payload)
    return submitExisting(saved.id)
  }

  async function deletePaper(id: string | number) {
    await AlibApi.deleteMySubmission(id)
    items.value = items.value.filter((paper) => paper.id !== String(id))
  }

  function resetForLogout() {
    items.value = []
    lastLoaded.value = null
  }

  return {
    items,
    isLoading,
    lastLoaded,
    papers,
    editablePaperIds,
    canEdit,
    canDelete,
    loadMyPapers,
    getMyPapers,
    getById,
    fetchSubmission,
    saveDraft,
    submitExisting,
    submitPaper,
    deletePaper,
    upsertPaper,
    resetForLogout,
  }
})
