import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { AlibApi } from '@/api/useAlibApi'
import type { AuthorProfileResponse, AuthorProfileUpdateRequest } from '@/api/types'

export const useAuthorProfileStore = defineStore('authorProfile', () => {
  const profile = ref<AuthorProfileResponse | null>(null)
  const isLoading = ref(false)
  const error = ref('')

  const hasConfirmedOrcid = computed(() => !!profile.value?.confirmed && !!profile.value?.orcid)

  async function loadProfile(force = false) {
    if (isLoading.value) return profile.value
    if (profile.value && !force) return profile.value
    isLoading.value = true
    error.value = ''
    try {
      profile.value = await AlibApi.getAuthorProfile()
      return profile.value
    } catch (e: any) {
      error.value = e?.message || 'Failed to load author profile'
      profile.value = null
      throw e
    } finally {
      isLoading.value = false
    }
  }

  async function saveProfile(payload: AuthorProfileUpdateRequest) {
    isLoading.value = true
    error.value = ''
    try {
      profile.value = await AlibApi.updateAuthorProfile(payload)
      return profile.value
    } catch (e: any) {
      error.value = e?.message || 'Failed to save author profile'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  async function removeProfile() {
    isLoading.value = true
    error.value = ''
    try {
      await AlibApi.deleteAuthorProfile()
      profile.value = {
        confirmed: false,
        paper_count: 0,
      }
    } catch (e: any) {
      error.value = e?.message || 'Failed to remove author profile'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  function reset() {
    profile.value = null
    error.value = ''
    isLoading.value = false
  }

  return {
    profile,
    isLoading,
    error,
    hasConfirmedOrcid,
    loadProfile,
    saveProfile,
    removeProfile,
    reset,
  }
})
