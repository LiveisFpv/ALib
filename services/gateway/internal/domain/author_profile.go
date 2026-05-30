package domain

type AuthorProfile struct {
	Orcid       string `json:"orcid"`
	Confirmed   bool   `json:"confirmed"`
	ConfirmedAt string `json:"confirmed_at"`
	PaperCount  int64  `json:"paper_count"`
}

type AuthorProfileUpdateInput struct {
	Orcid             string `json:"orcid"`
	ConfirmAuthorship bool   `json:"confirm_authorship"`
}
