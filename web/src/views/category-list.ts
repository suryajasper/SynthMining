import m from 'mithril';
import '../css/tags.scss';
import { fetchRequest, initArr } from '../utils/utils';
import { icons } from './icons';
import { TagAttrs } from './project-loader';

interface CategoryViewAttrs {
  isAdmin: boolean;
  tagName: TagAttrs;
  selected: boolean;
  highlighted: number;
  color: string;
  applyToImageMode: boolean;

  select: () => void;
  applyToImage: () => void;
  activatePopup: () => void;
}

const CategoryView : m.Component<CategoryViewAttrs> = {
  view({attrs}) {
    const tag = attrs.tagName;

    return m('div.category-item', {
      class: [
        attrs.selected ? 'selected show-category-description' : '',
        attrs.highlighted ? 'show-category-description' : '',
      ].join(' '),

      style: {
        '--highlight-fill': attrs.highlighted,
      },

      onclick: (e: MouseEvent) => {
        if (attrs.applyToImageMode)
          attrs.applyToImage();
        else
          attrs.select();
      },
    }, [
      m('div.category-header', [
        m('span.category-item-color', {
          style: {
            backgroundColor: attrs.color,
          }
        }),
        m('span.category-item-title', tag.name),
      ]),
      
      m('div.category-description', tag.description),

      (attrs.applyToImageMode || !attrs.isAdmin) ? null : 
        m('div.hover-menu', 
          m('button.tag-edit-button', {
            onclick: e => {
              e.stopPropagation();

              attrs.activatePopup();
            }
          }, 
            icons.gear,
          )
        )
    ]);
  }
};

export interface CategorySelectionsAttrs {
  uid: string | undefined;
  isAdmin: boolean;
  projectId: string;

  categories: TagAttrs[];
  activeCategories: Record<string, number>;

  addTag: () => void;
  updateTag: (tagId: string, update: {
    name: string,
    description: string,
    goalQty: number,
  }) => void;
  removeTag: (tagId: string) => void;

  updatePopupStatus: (state: {
    name: string,
    active: boolean,
    data: object,
  }) => void;
  unshiftSelection: (i: number) => void;
  applyToImage: (tag: TagAttrs) => void;

  applyToImageMode: boolean;
}

export class CategorySelections implements m.ClassComponent<CategorySelectionsAttrs> {
  private selected: boolean[];

  constructor(vnode: m.CVnode<CategorySelectionsAttrs>) {
    this.selected = [];
  }

  createCategoryItem(
    attrs: CategorySelectionsAttrs,
    i: number,
  ) : m.Vnode<CategoryViewAttrs> {
    const tag = attrs.categories[i];
    let isHighlighted = attrs.activeCategories[tag._id];

    return m(CategoryView, {
      isAdmin: attrs.isAdmin,
      tagName: tag,
      selected: attrs.applyToImageMode ? false : this.selected[i],
      highlighted: isHighlighted,
      color: attrs.categories[i].color,
      applyToImageMode: attrs.applyToImageMode,

      select: () => {
        if (!this.selected[i]) {
          attrs.unshiftSelection(i);
          for (let j = this.selected.length-1; j >= 1; j--)
            this.selected[j] = this.selected[j-1];
          this.selected[0] = true;
        } else {
          this.selected[i] = false;
        }
      },

      applyToImage: () => {
        attrs.applyToImage(attrs.categories[i]);
      },

      activatePopup: () => {
        console.log('activating edit tag popup with', tag);
        attrs.updatePopupStatus({
          name: 'editTag', 
          active: true,
          data: {
            projectId: attrs.projectId,
            uid: attrs.uid,
            tag,

            updateTag: attrs.updateTag,
            removeTag: attrs.removeTag,
          },
        });
      },
    })
  }

  view({attrs}: m.CVnode<CategorySelectionsAttrs>) {
    if (this.selected.length != attrs.categories.length)
      this.selected = initArr<boolean>(attrs.categories.length, false);
    
    return m('div.category-container', [
      attrs.applyToImageMode ?
        m('span.category-list-header', 'Click Tags to Apply to Images') : null,

      m('div.category-content',
        (attrs.categories || [])
          .map((_, i) => 
            this.createCategoryItem(attrs, i)
          )
          .concat((attrs.applyToImageMode || !attrs.isAdmin) ? 
            [] : [
              m('div.category-item.new-category-button.selected', {
                onclick: attrs.addTag,
              }, [
                m('span.category-item-new-icon'),
                m('span.category-item-title', 'Add Tag'),
              ])
            ]
          )
      ),
    ]);
  }
}