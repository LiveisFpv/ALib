package presenters

type AuthorProfileResponse struct {
	Orcid       string `json:"orcid,omitempty"`
	Confirmed   bool   `json:"confirmed"`
	ConfirmedAt string `json:"confirmed_at,omitempty"`
	PaperCount  int64  `json:"paper_count"`
}

type AuthorProfileUpdateRequest struct {
	Orcid             string `json:"orcid" binding:"required"`
	ConfirmAuthorship bool   `json:"confirm_authorship"`
}
