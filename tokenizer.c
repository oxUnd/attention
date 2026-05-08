#include "tokenizer.h"
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SPECIALS[TOK_NUM_SPECIALS] = {"<pad>", "<unk>", "<bos>", "<eos>"};

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) {
        memcpy(p, s, n);
    }
    return p;
}

static unsigned long hash_str(const char *s) {
    unsigned long h = 1469598103934665603UL;
    while (*s) {
        h ^= (unsigned char)(*s++);
        h *= 1099511628211UL;
    }
    return h;
}

static void hash_init(Tokenizer *t, int n_buckets) {
    t->n_buckets = n_buckets;
    t->buckets = (TokenizerEntry **)calloc((size_t)n_buckets, sizeof(TokenizerEntry *));
}

static void hash_free(Tokenizer *t) {
    if (!t->buckets) return;
    for (int i = 0; i < t->n_buckets; i++) {
        TokenizerEntry *e = t->buckets[i];
        while (e) {
            TokenizerEntry *next = e->next;
            free(e->token);
            free(e);
            e = next;
        }
    }
    free(t->buckets);
    t->buckets = NULL;
    t->n_buckets = 0;
}

static int hash_lookup(const Tokenizer *t, const char *token) {
    if (!t->buckets || !t->n_buckets) return -1;
    unsigned long h = hash_str(token) % (unsigned long)t->n_buckets;
    TokenizerEntry *e = t->buckets[h];
    while (e) {
        if (strcmp(e->token, token) == 0) return e->id;
        e = e->next;
    }
    return -1;
}

static void hash_insert(Tokenizer *t, const char *token, int id) {
    unsigned long h = hash_str(token) % (unsigned long)t->n_buckets;
    TokenizerEntry *e = (TokenizerEntry *)malloc(sizeof(TokenizerEntry));
    e->token = xstrdup(token);
    e->id = id;
    e->next = t->buckets[h];
    t->buckets[h] = e;
}

static void install_specials(Tokenizer *t) {
    for (int i = 0; i < TOK_NUM_SPECIALS; i++) {
        t->id_to_token[i] = xstrdup(SPECIALS[i]);
        hash_insert(t, SPECIALS[i], i);
    }
}

Tokenizer *tokenizer_create_byte(void) {
    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->kind = TOKENIZER_BYTE;
    t->vocab_size = TOK_NUM_SPECIALS + 256;
    t->max_token_len = 1;
    t->lowercase = 0;
    t->id_to_token = (char **)calloc((size_t)t->vocab_size, sizeof(char *));
    hash_init(t, t->vocab_size * 2 + 7);
    install_specials(t);
    for (int i = 0; i < 256; i++) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\x%02x", i);
        int id = TOK_NUM_SPECIALS + i;
        t->id_to_token[id] = xstrdup(buf);
    }
    for (int i = 0; i < 256; i++) {
        t->char_to_id[i] = TOK_NUM_SPECIALS + i;
    }
    return t;
}

Tokenizer *tokenizer_create_char(const char *charset, int lowercase) {
    if (!charset) charset = "";
    int n = (int)strlen(charset);
    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->kind = TOKENIZER_CHAR;
    t->vocab_size = TOK_NUM_SPECIALS + n;
    t->max_token_len = 1;
    t->lowercase = lowercase ? 1 : 0;
    t->id_to_token = (char **)calloc((size_t)t->vocab_size, sizeof(char *));
    hash_init(t, t->vocab_size * 2 + 7);
    install_specials(t);

    for (int i = 0; i < 256; i++) t->char_to_id[i] = -1;

    for (int i = 0; i < n; i++) {
        char buf[2] = {charset[i], 0};
        int id = TOK_NUM_SPECIALS + i;
        t->id_to_token[id] = xstrdup(buf);
        hash_insert(t, buf, id);
        t->char_to_id[(unsigned char)charset[i]] = id;
        if (lowercase) {
            char c = charset[i];
            if (c >= 'a' && c <= 'z') {
                t->char_to_id[(unsigned char)(c - 32)] = id;
            }
        }
    }
    return t;
}

typedef struct {
    char *word;
    int count;
} WordCount;

static int wc_cmp_desc(const void *a, const void *b) {
    const WordCount *x = (const WordCount *)a;
    const WordCount *y = (const WordCount *)b;
    if (y->count != x->count) return y->count - x->count;
    return strcmp(x->word, y->word);
}

static int is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '\'';
}

static int is_punct(char c) {
    return c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':' || c == '-' || c == '"';
}

typedef struct {
    char **tokens;
    int count;
    int cap;
} TokList;

static void toklist_push(TokList *tl, const char *token) {
    if (tl->count >= tl->cap) {
        tl->cap = tl->cap ? tl->cap * 2 : 64;
        tl->tokens = (char **)realloc(tl->tokens, (size_t)tl->cap * sizeof(char *));
    }
    tl->tokens[tl->count++] = xstrdup(token);
}

static void toklist_free(TokList *tl) {
    for (int i = 0; i < tl->count; i++) free(tl->tokens[i]);
    free(tl->tokens);
    tl->tokens = NULL;
    tl->count = 0;
    tl->cap = 0;
}

static void word_tokenize(const char *text, int lowercase, TokList *out) {
    char buf[256];
    int bi = 0;
    for (size_t i = 0; text[i]; i++) {
        char c = text[i];
        if (lowercase && c >= 'A' && c <= 'Z') c = c + 32;
        if (is_word_char(c)) {
            if (bi < (int)sizeof(buf) - 1) buf[bi++] = c;
        } else {
            if (bi > 0) {
                buf[bi] = 0;
                toklist_push(out, buf);
                bi = 0;
            }
            if (is_punct(c)) {
                char p[2] = {c, 0};
                toklist_push(out, p);
            }
        }
    }
    if (bi > 0) {
        buf[bi] = 0;
        toklist_push(out, buf);
    }
}

Tokenizer *tokenizer_create_word(const char *corpus, int max_vocab, int lowercase) {
    if (!corpus) return NULL;
    if (max_vocab < TOK_NUM_SPECIALS + 1) max_vocab = TOK_NUM_SPECIALS + 1;

    TokList tokens = {0};
    word_tokenize(corpus, lowercase, &tokens);

    int wc_cap = 64;
    int wc_n = 0;
    WordCount *wcs = (WordCount *)malloc((size_t)wc_cap * sizeof(WordCount));
    int n_buckets = 1024;
    Tokenizer tmp = {0};
    tmp.n_buckets = n_buckets;
    tmp.buckets = (TokenizerEntry **)calloc((size_t)n_buckets, sizeof(TokenizerEntry *));

    for (int i = 0; i < tokens.count; i++) {
        const char *tok = tokens.tokens[i];
        int idx = -1;
        unsigned long h = hash_str(tok) % (unsigned long)n_buckets;
        TokenizerEntry *e = tmp.buckets[h];
        while (e) {
            if (strcmp(e->token, tok) == 0) {
                idx = e->id;
                break;
            }
            e = e->next;
        }
        if (idx < 0) {
            if (wc_n >= wc_cap) {
                wc_cap *= 2;
                wcs = (WordCount *)realloc(wcs, (size_t)wc_cap * sizeof(WordCount));
            }
            wcs[wc_n].word = xstrdup(tok);
            wcs[wc_n].count = 1;
            TokenizerEntry *ne = (TokenizerEntry *)malloc(sizeof(TokenizerEntry));
            ne->token = xstrdup(tok);
            ne->id = wc_n;
            ne->next = tmp.buckets[h];
            tmp.buckets[h] = ne;
            wc_n++;
        } else {
            wcs[idx].count++;
        }
    }

    for (int i = 0; i < tmp.n_buckets; i++) {
        TokenizerEntry *e = tmp.buckets[i];
        while (e) {
            TokenizerEntry *next = e->next;
            free(e->token);
            free(e);
            e = next;
        }
    }
    free(tmp.buckets);
    toklist_free(&tokens);

    qsort(wcs, (size_t)wc_n, sizeof(WordCount), wc_cmp_desc);

    int keep = wc_n + TOK_NUM_SPECIALS;
    if (keep > max_vocab) keep = max_vocab;

    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->kind = TOKENIZER_WORD;
    t->vocab_size = keep;
    t->lowercase = lowercase ? 1 : 0;
    t->id_to_token = (char **)calloc((size_t)keep, sizeof(char *));
    hash_init(t, keep * 2 + 7);
    install_specials(t);

    int max_len = 1;
    for (int i = 0; i < keep - TOK_NUM_SPECIALS && i < wc_n; i++) {
        int id = TOK_NUM_SPECIALS + i;
        t->id_to_token[id] = xstrdup(wcs[i].word);
        hash_insert(t, wcs[i].word, id);
        int L = (int)strlen(wcs[i].word);
        if (L > max_len) max_len = L;
    }
    t->max_token_len = max_len;

    for (int i = 0; i < 256; i++) t->char_to_id[i] = -1;

    for (int i = 0; i < wc_n; i++) free(wcs[i].word);
    free(wcs);
    return t;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    if (t->id_to_token) {
        for (int i = 0; i < t->vocab_size; i++) free(t->id_to_token[i]);
        free(t->id_to_token);
    }
    hash_free(t);
    free(t);
}

const char *tokenizer_id_to_token(const Tokenizer *t, int id) {
    if (!t || id < 0 || id >= t->vocab_size) return "";
    return t->id_to_token[id];
}

int tokenizer_token_to_id(const Tokenizer *t, const char *token) {
    if (!t || !token) return TOK_UNK;
    int id = hash_lookup(t, token);
    return id >= 0 ? id : TOK_UNK;
}

int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_out) {
    if (!t || !text || !out) return 0;
    int n = 0;
    if (t->kind == TOKENIZER_BYTE) {
        for (size_t i = 0; text[i] && n < max_out; i++) {
            out[n++] = t->char_to_id[(unsigned char)text[i]];
        }
        return n;
    }
    if (t->kind == TOKENIZER_CHAR) {
        for (size_t i = 0; text[i] && n < max_out; i++) {
            unsigned char c = (unsigned char)text[i];
            int id = t->char_to_id[c];
            if (id < 0 && t->lowercase && c >= 'A' && c <= 'Z') {
                id = t->char_to_id[c + 32];
            }
            if (id < 0) id = TOK_UNK;
            out[n++] = id;
        }
        return n;
    }
    /* word */
    TokList tl = {0};
    word_tokenize(text, t->lowercase, &tl);
    for (int i = 0; i < tl.count && n < max_out; i++) {
        out[n++] = tokenizer_token_to_id(t, tl.tokens[i]);
    }
    toklist_free(&tl);
    return n;
}

int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int max_out) {
    if (!t || !ids || !out || max_out <= 0) return 0;
    int pos = 0;
    out[0] = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size) continue;
        if (id < TOK_NUM_SPECIALS) {
            if (id == TOK_PAD) continue;
            if (id == TOK_BOS) continue;
            if (id == TOK_EOS) break;
        }
        const char *tok = t->id_to_token[id];
        if (!tok) continue;
        int L = (int)strlen(tok);
        int need_space = 0;
        if (t->kind == TOKENIZER_WORD && pos > 0 && L > 0) {
            char first = tok[0];
            if (!is_punct(first)) need_space = 1;
        }
        if (need_space && pos + 1 < max_out) {
            out[pos++] = ' ';
        }
        for (int j = 0; j < L && pos + 1 < max_out; j++) {
            out[pos++] = tok[j];
        }
        out[pos] = 0;
    }
    return pos;
}

void tokenizer_print_stats(const Tokenizer *t, FILE *out) {
    if (!t || !out) return;
    const char *kind = "?";
    switch (t->kind) {
        case TOKENIZER_CHAR: kind = "char"; break;
        case TOKENIZER_BYTE: kind = "byte"; break;
        case TOKENIZER_WORD: kind = "word"; break;
    }
    fprintf(out, "Tokenizer{kind=%s, vocab=%d, max_token_len=%d, lowercase=%d}\n",
            kind, t->vocab_size, t->max_token_len, t->lowercase);
}

int tokenizer_save(const Tokenizer *t, const char *path) {
    if (!t || !path) return -1;
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "tokenizer_v1\n");
    fprintf(f, "kind %d\n", (int)t->kind);
    fprintf(f, "vocab %d\n", t->vocab_size);
    fprintf(f, "lowercase %d\n", t->lowercase);
    fprintf(f, "max_token_len %d\n", t->max_token_len);
    for (int i = 0; i < t->vocab_size; i++) {
        const char *s = t->id_to_token[i] ? t->id_to_token[i] : "";
        fprintf(f, "%d\t", i);
        for (size_t k = 0; s[k]; k++) {
            char c = s[k];
            if (c == '\\') fputs("\\\\", f);
            else if (c == '\n') fputs("\\n", f);
            else if (c == '\t') fputs("\\t", f);
            else fputc(c, f);
        }
        fputc('\n', f);
    }
    fclose(f);
    return 0;
}

static char *unescape(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\\' && i + 1 < n) {
            char nx = s[i + 1];
            if (nx == '\\') {
                out[j++] = '\\';
                i++;
            } else if (nx == 'n') {
                out[j++] = '\n';
                i++;
            } else if (nx == 't') {
                out[j++] = '\t';
                i++;
            } else {
                out[j++] = s[i];
            }
        } else {
            out[j++] = s[i];
        }
    }
    out[j] = 0;
    return out;
}

Tokenizer *tokenizer_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    char line[4096];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return NULL;
    }
    if (strncmp(line, "tokenizer_v1", 12) != 0) {
        fclose(f);
        return NULL;
    }
    int kind = 0, vocab = 0, lowercase = 0, max_token_len = 1;
    if (fscanf(f, "kind %d\n", &kind) != 1) { fclose(f); return NULL; }
    if (fscanf(f, "vocab %d\n", &vocab) != 1) { fclose(f); return NULL; }
    if (fscanf(f, "lowercase %d\n", &lowercase) != 1) { fclose(f); return NULL; }
    if (fscanf(f, "max_token_len %d\n", &max_token_len) != 1) { fclose(f); return NULL; }

    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->kind = (TokenizerKind)kind;
    t->vocab_size = vocab;
    t->lowercase = lowercase;
    t->max_token_len = max_token_len;
    t->id_to_token = (char **)calloc((size_t)vocab, sizeof(char *));
    hash_init(t, vocab * 2 + 7);
    for (int i = 0; i < 256; i++) t->char_to_id[i] = -1;

    for (int i = 0; i < vocab; i++) {
        if (!fgets(line, sizeof(line), f)) {
            tokenizer_free(t);
            fclose(f);
            return NULL;
        }
        size_t L = strlen(line);
        if (L && line[L - 1] == '\n') line[--L] = 0;
        char *tab = strchr(line, '\t');
        int id = i;
        const char *raw_tok = line;
        if (tab) {
            *tab = 0;
            id = atoi(line);
            raw_tok = tab + 1;
        }
        char *tok = unescape(raw_tok);
        t->id_to_token[id] = tok;
        hash_insert(t, tok, id);
        if (t->kind == TOKENIZER_CHAR && id >= TOK_NUM_SPECIALS) {
            unsigned char c = (unsigned char)tok[0];
            t->char_to_id[c] = id;
            if (lowercase && c >= 'a' && c <= 'z') {
                t->char_to_id[c - 32] = id;
            }
        }
        if (t->kind == TOKENIZER_BYTE && id >= TOK_NUM_SPECIALS) {
            int b = id - TOK_NUM_SPECIALS;
            if (b >= 0 && b < 256) t->char_to_id[b] = id;
        }
    }
    fclose(f);
    return t;
}
